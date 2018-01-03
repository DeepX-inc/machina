import torch
import torch.nn as nn
import numpy as np

from machina.utils import Variable
from machina.misc import logger

def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def linesearch(
        pol,
        batch,
        f,
        x,
        fullstep,
        expected_improve_rate,
        max_backtracks=10,
        accept_ratio=.1
    ):
    fval = f(pol, batch, True).data
    print("loss before", fval[0])
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        nn.utils.vector_to_parameters(Variable(xnew), pol.parameters())
        newfval = f(pol, batch, True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        print("a/e/r", actual_improve[0], expected_improve[0], ratio[0])

        if ratio[0] > accept_ratio and actual_improve[0] > 0:
        #if actual_improve[0] > 0:
            print("loss after", newfval[0])
            return True, xnew
    return False, x

def make_pol_loss(pol, batch, volatile=False):
    obs = Variable(batch['obs'], volatile=volatile)
    acs = Variable(batch['acs'], volatile=volatile)
    advs = Variable(batch['advs'], volatile=volatile)
    _, _, pd_params = pol(obs)
    llh = pol.pd.llh(acs, pd_params['mean'], pd_params['log_std'])

    pol_loss = - torch.mean(llh * advs)
    return pol_loss

def make_kl(pol, batch):
    obs = Variable(batch['obs'])
    _, _, pd_params = pol(obs)
    return pol.pd.kl_pq(
        Variable(pd_params['mean'].data), Variable(pd_params['log_std'].data),
        pd_params['mean'], pd_params['log_std']
    )

def update_pol(pol, batch, make_pol_loss=make_pol_loss, make_kl=make_kl, max_kl=0.01, damping=0.1):
    pol_loss = make_pol_loss(pol, batch)
    grads = torch.autograd.grad(pol_loss, pol.parameters(), create_graph=True)
    flat_pol_loss_grad = nn.utils.parameters_to_vector(grads).data

    def Fvp(v):
        kl = make_kl(pol, batch)
        kl = torch.mean(kl)

        grads = torch.autograd.grad(kl, pol.parameters(), create_graph=True)
        flat_grad_kl = nn.utils.parameters_to_vector(grads)
        gvp = torch.sum(flat_grad_kl * Variable(v))
        grads = torch.autograd.grad(gvp, pol.parameters())
        grads = [g.contiguous() for g in grads]
        fvp = nn.utils.parameters_to_vector(grads).data

        return fvp + v * damping

    stepdir = conjugate_gradients(Fvp, -flat_pol_loss_grad, 10)

    shs = 0.5 * torch.sum(stepdir * Fvp(stepdir), 0, keepdim=True)

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdotstepdir = torch.sum(-flat_pol_loss_grad * stepdir, 0, keepdim=True)

    prev_params = nn.utils.parameters_to_vector(pol.parameters()).data
    success, new_params = linesearch(pol, batch, make_pol_loss, prev_params, fullstep,
                                     neggdotstepdir / lm[0])
    nn.utils.vector_to_parameters(Variable(new_params), pol.parameters())

    return pol_loss.data.cpu().numpy()

def make_vf_loss(vf, batch):
    obs = Variable(batch['obs'])
    rets = Variable(batch['rets'])
    vf_loss = 0.5 * torch.mean((vf(obs) - rets)**2)
    return vf_loss

def update_vf(vf, optim_vf, batch):
    vf_loss = make_vf_loss(vf, batch)
    optim_vf.zero_grad()
    vf_loss.backward()
    optim_vf.step()
    return vf_loss.data.cpu().numpy()

def train(data, pol, vf,
        optim_vf,
        epoch=5, batch_size=64,# optimization hypers
        ):

    pol_losses = []
    vf_losses = []
    logger.log("Optimizing...")
    for batch in data.full_batch(epoch):
        pol_loss = update_pol(pol, batch)
        pol_losses.append(pol_loss)
        #vf.set_mean(torch.mean(batch['rets'], 0, keepdim=True))
        #vf.set_std(torch.std(batch['rets'], 0, keepdim=True))

    for batch in data.iterate(batch_size=64, epoch=5):
        vf_loss = update_vf(vf, optim_vf, batch)
        vf_losses.append(vf_loss)

    logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses, VfLoss=vf_losses)



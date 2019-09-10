"""
This is an implementation of Trust Region Policy Optimization.
See https://arxiv.org/abs/1502.05477
"""

import torch
import torch.nn as nn
import numpy as np

from machina import loss_functional as lf
from machina.utils import detach_tensor_dict
from machina import logger


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    """
    Calculating conjugate gradient
    """
    x = torch.zeros_like(b)
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
    accept_ratio=.1,
    ent_beta=0
):
    fval = f(pol, batch).detach()
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        nn.utils.vector_to_parameters(xnew, pol.parameters())
        newfval = f(pol, batch, ent_beta=ent_beta).detach()
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            return True, xnew
    return False, x


def make_kl(pol, batch):
    obs = batch['obs']

    pol.reset()
    if pol.rnn:
        h_masks = batch['h_masks']
        out_masks = batch['out_masks']
        _, _, pd_params = pol(obs, h_masks=h_masks)
    else:
        out_masks = torch.ones_like(batch['advs'])
        _, _, pd_params = pol(obs)

    return pol.pd.kl_pq(
        detach_tensor_dict(pd_params),
        pd_params
    )


def update_pol(pol, batch, make_kl=make_kl, max_kl=0.01, damping=0.1, num_cg=10, ent_beta=0):
    pol_loss = lf.pg(pol, batch, ent_beta)
    grads = torch.autograd.grad(pol_loss, pol.parameters(), create_graph=True)
    grads = [g.contiguous() for g in grads]
    flat_pol_loss_grad = nn.utils.parameters_to_vector(grads).detach()

    def Fvp(v):
        kl = make_kl(pol, batch)
        kl = torch.mean(kl)

        grads = torch.autograd.grad(kl, pol.parameters(), create_graph=True)
        grads = [g.contiguous() for g in grads]
        flat_grad_kl = nn.utils.parameters_to_vector(grads)
        gvp = torch.sum(flat_grad_kl * v)
        grads = torch.autograd.grad(gvp, pol.parameters())
        grads = [g.contiguous() for g in grads]
        fvp = nn.utils.parameters_to_vector(grads).detach()

        return fvp + v * damping

    stepdir = conjugate_gradients(Fvp, -flat_pol_loss_grad, num_cg)

    shs = 0.5 * torch.sum(stepdir * Fvp(stepdir), 0, keepdim=True)
    if (shs < 0).any():
        logger.log('invalid shs')
        return pol_loss.data.cpu().numpy()

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdotstepdir = torch.sum(-flat_pol_loss_grad * stepdir, 0, keepdim=True)

    prev_params = nn.utils.parameters_to_vector(
        [p.contiguous() for p in pol.parameters()]).detach()
    success, new_params = linesearch(pol, batch, lf.pg, prev_params, fullstep,
                                     neggdotstepdir / lm[0], ent_beta=ent_beta)
    nn.utils.vector_to_parameters(new_params, pol.parameters())

    return pol_loss.detach().cpu().numpy()


def update_vf(vf, optim_vf, batch):
    vf_loss = lf.monte_carlo(vf, batch)
    optim_vf.zero_grad()
    vf_loss.backward()
    optim_vf.step()
    return vf_loss.detach().cpu().numpy()


def train(traj, pol, vf,
          optim_vf,
          epoch=5, batch_size=64, num_epi_per_seq=1,  # optimization hypers
          max_kl=0.01, num_cg=10, damping=0.1, ent_beta=0,
          log_enable=True,
          ):
    """
    Train function for trust region policy optimization.

    Parameters
    ----------
    traj : Traj
        On policy trajectory.
    pol : Pol
        Policy.
    vf : SVfunction
        V function.
    optim_vf : torch.optim.Optimizer
        Optimizer for V function.
    epoch : int
        Number of iteration.
    batch_size : int
        Number of batches.
    num_epi_per_seq : int
        Number of episodes in one sequence for rnn.
    max_kl : float
        Limit of KL divergence.
    num_cg : int
        Number of iteration in conjugate gradient computation.
    damping : float
        Damping parameter for Hessian Vector Product.
    log_enable: bool
        If True, enable logging

    Returns
    -------
    result_dict : dict
        Dictionary which contains losses information.
    """

    pol_losses = []
    vf_losses = []
    if log_enable:
        logger.log("Optimizing...")
    iterator = traj.full_batch(1) if not pol.rnn else traj.iterate_rnn(
        batch_size=traj.num_epi)
    for batch in iterator:
        pol_loss = update_pol(pol, batch, max_kl=max_kl,
                              num_cg=num_cg, damping=damping, ent_beta=ent_beta)
        pol_losses.append(pol_loss)

    iterator = traj.iterate(batch_size, epoch) if not pol.rnn else traj.iterate_rnn(
        batch_size=batch_size, num_epi_per_seq=num_epi_per_seq, epoch=epoch)
    for batch in iterator:
        vf_loss = update_vf(vf, optim_vf, batch)
        vf_losses.append(vf_loss)

    if log_enable:
        logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses, VfLoss=vf_losses)

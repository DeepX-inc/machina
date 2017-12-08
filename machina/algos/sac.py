import torch
import torch.nn as nn
from machina.utils import Variable, torch2torch
from machina.misc import logger

def make_pol_loss(pol, qf, vf, batch, sampling):
    obs = Variable(batch['obs'])

    pol_loss = 0
    _, _, pd_params = pol(obs)
    means, log_stds = pd_params['mean'], pd_params['log_std']
    for _ in range(sampling):
        acs = means + Variable(torch2torch(torch.randn(means.size()))) * torch.exp(log_stds)
        llh = pol.pd.llh(Variable(acs.data), pd_params['mean'], pd_params['log_std'])
        pol_loss += llh * Variable(llh.data - qf(obs, acs).data + vf(obs).data)
    pol_loss /= sampling

    return torch.mean(pol_loss)

def make_qf_loss(qf, vf, batch, gamma):
    obs = Variable(batch['obs'])
    acs = Variable(batch['acs'])
    rews = Variable(batch['rews'])
    next_obs = Variable(batch['next_obs'])
    terminals = Variable(batch['terminals'])

    targ = rews + gamma * vf(next_obs) * (1 - terminals)
    targ = Variable(targ.data)

    return 0.5 * torch.mean((qf(obs, acs) - targ)**2)

def make_vf_loss(pol, qf, vf, batch, sampling):
    obs = Variable(batch['obs'])

    targ = 0
    _, _, pd_params = pol(obs)
    means, log_stds = pd_params['mean'], pd_params['log_std']
    for _ in range(sampling):
        acs = means + Variable(torch2torch(torch.randn(means.size()))) * torch.exp(log_stds)
        llh = pol.pd.llh(acs, pd_params['mean'], pd_params['log_std'])
        targ += qf(obs, acs) - llh
    targ /= sampling
    targ = Variable(targ.data)

    return 0.5 * torch.mean((vf(obs) - targ)**2)

def train(off_data,
        pol, qf, vf,
        optim_pol, optim_qf, optim_vf,
        epoch, batch_size,# optimization hypers
        gamma, sampling,
        ):

    vf_losses = []
    qf_losses = []
    pol_losses = []
    logger.log("Optimizing...")
    for batch in off_data.iterate(batch_size, epoch):
        vf_loss = make_vf_loss(pol, qf, vf, batch, sampling)
        optim_vf.zero_grad()
        vf_loss.backward()
        optim_vf.step()

        qf_loss = make_qf_loss(qf, vf, batch, gamma)
        optim_qf.zero_grad()
        qf_loss.backward()
        optim_qf.step()

        pol_loss = make_pol_loss(pol, qf, vf, batch, sampling)
        optim_pol.zero_grad()
        pol_loss.backward()
        optim_pol.step()

        vf_losses.append(vf_loss.data.cpu().numpy())
        qf_losses.append(qf_loss.data.cpu().numpy())
        pol_losses.append(pol_loss.data.cpu().numpy())

    logger.log("Optimization finished!")

    return dict(
        VfLoss=vf_losses,
        QfLoss=qf_losses,
        PolLoss=pol_losses,
    )


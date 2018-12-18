

import torch
import torch.nn as nn
from machina.misc import logger


def make_pol_loss(pol, qf, batch):
    obs = batch['obs']
    _, _, param = pol(obs)
    q, _ = qf(obs, param['mean'])
    pol_loss = -torch.mean(q)
    return pol_loss


def make_bellman_loss(qf, targ_qf, targ_pol, batch, gamma):
    obs = batch['obs']
    acs = batch['acs']
    rews = batch['rews']
    next_obs = batch['next_obs']
    dones = batch['dones']
    _, _, param = targ_pol(next_obs)
    next_q, _ = targ_qf(next_obs, param['mean'])
    targ = rews + gamma * next_q * (1 - dones)
    targ = targ.detach()
    q, _ = qf(obs, acs)
    return 0.5 * torch.mean((q - targ)**2)


def train(traj,
        pol, targ_pol, qf, targ_qf,
        optim_pol, optim_qf,
        epoch, batch_size,# optimization hypers
        tau, gamma, lam # advantage estimation
        ):

    pol_losses = []
    qf_losses = []
    logger.log("Optimizing...")
    for batch in traj.random_batch(batch_size, epoch):
        qf_bellman_loss = make_bellman_loss(qf, targ_qf, targ_pol, batch, gamma)
        optim_qf.zero_grad()
        qf_bellman_loss.backward()
        optim_qf.step()

        pol_loss = make_pol_loss(pol, qf, batch)
        optim_pol.zero_grad()
        pol_loss.backward()
        optim_pol.step()

        for q, targ_q, p, targ_p in zip(qf.parameters(), targ_qf.parameters(), pol.parameters(), targ_pol.parameters()):
            targ_p.detach().copy_((1 - tau) * targ_p.detach() + tau * p.detach())
            targ_q.detach().copy_((1 - tau) * targ_q.detach() + tau * q.detach())
        qf_losses.append(qf_bellman_loss.detach().cpu().numpy())
        pol_losses.append(pol_loss.detach().cpu().numpy())
    logger.log("Optimization finished!")

    return {'PolLoss': pol_losses, 'QfLoss': qf_losses}

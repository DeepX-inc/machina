

import torch
import torch.nn as nn
from machina.utils import Variable, torch2torch, np2torch
from machina.misc import logger


def make_pol_loss(pol, qf, batch):
    obs = Variable(batch['obs'].float())
    q = 0
    _, _, param = pol(obs)
    q = qf(obs, param['mean'])
    pol_loss = -torch.mean(q)
    return pol_loss


def make_bellman_loss(qf, targ_qf, targ_pol, batch, gamma):
    obs = Variable(batch['obs'].float())
    acs = Variable(batch['acs'].float())
    rews = Variable(batch['rews'].float())
    next_obs = Variable(batch['next_obs'].float())
    terminals = Variable(batch['terminals'].float())
    next_q = 0
    _, _, param = targ_pol(next_obs)
    next_q += targ_qf(next_obs, param['mean'])
    targ = rews + gamma * next_q * (1 - terminals)
    targ = Variable(targ.data)
    return 0.5 * torch.mean((qf(obs, acs) - targ)**2)


def train(off_data,
        pol, targ_pol, qf, targ_qf,
        optim_pol, optim_qf,
        epoch, batch_size,# optimization hypers
        tau, gamma, lam # advantage estimation
        ):

    pol_losses = []
    qf_losses = []
    logger.log("Optimizing...")
    for batch in off_data.iterate(batch_size, epoch):
        qf_bellman_loss = make_bellman_loss(qf, targ_qf, targ_pol, batch, gamma)
        optim_qf.zero_grad()
        qf_bellman_loss.backward()
        optim_qf.step()

        pol_loss = make_pol_loss(pol, qf, batch)
        optim_pol.zero_grad()
        pol_loss.backward()
        optim_pol.step()

        for q, targ_q, p, targ_p in zip(qf.parameters(), targ_qf.parameters(), pol.parameters(), targ_pol.parameters()):
            targ_p.data.copy_((1 - tau) * targ_p.data + tau * p.data)
            targ_q.data.copy_((1 - tau) * targ_q.data + tau * q.data)
        qf_losses.append(qf_bellman_loss.data.cpu().numpy())
        pol_losses.append(pol_loss.data.cpu().numpy())

    logger.log("Optimization finished!")

    return {'PolLoss': pol_losses, 'QfLoss': qf_losses}


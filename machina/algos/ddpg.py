# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:11:50 2017

@author: yoshi
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from ..misc import logger

def make_pol_loss(pol, qf, batch, sampling):
    obs = Variable(torch.from_numpy(batch['obs']).float())

    q = 0
    _, acs, _ = pol(obs)
    q=qf(obs,acs)
    pol_loss = -torch.mean(q)
#    for _ in range(sampling):
#        _, acs, _ = pol(obs)
#        q += qf(obs, acs)
#    q /= sampling

#    _, _, pd_params = pol(obs)
#    kl = pol.pd.kl_pq(
#        Variable(pd_params['mean'].data),
#        Variable(pd_params['log_std'].data),
#        pd_params['mean'],
#        pd_params['log_std']
#    )
#    mean_kl = torch.mean(kl)

    return pol_loss #+ mean_kl

def make_bellman_loss(qf, targ_qf, targ_pol, batch, gamma, sampling):
    obs = Variable(torch.from_numpy(batch['obs']).float())
    acs = Variable(torch.from_numpy(batch['acs']).float())
    rews = Variable(torch.from_numpy(batch['rews']).float())
    next_obs = Variable(torch.from_numpy(batch['next_obs']).float())
    terminals = Variable(torch.from_numpy(batch['terminals']).float())
    expected_next_q = 0
    for _ in range(sampling): #これいるのか
        _, next_acs, _ = targ_pol(next_obs) #このポリシーはノイズ込みだがそれでいいのか？
        expected_next_q += targ_qf(next_obs, next_acs) 
    expected_next_q /= sampling
    targ = rews + gamma * expected_next_q * (1 - terminals)
    targ = Variable(targ.data)

    return 0.5 * torch.mean((qf(obs, acs) - targ)**2)


def train(off_data,
        pol,targ_pol, qf, targ_qf,
        optim_pol, optim_qf,
        epoch, batch_size,# optimization hypers
        tau, gamma, lam, # advantage estimation
        sampling,
        ):

    pol_losses = []
    qf_losses = []
    logger.log("Optimizing...")
    for batch in off_data.iterate(batch_size, epoch):
        qf_bellman_loss = make_bellman_loss(qf, targ_qf, targ_pol, batch, gamma, sampling)
        optim_qf.zero_grad()
        qf_bellman_loss.backward()
        optim_qf.step()

        pol_loss = make_pol_loss(pol, qf, batch, sampling)
        optim_pol.zero_grad()
        pol_loss.backward()
        optim_pol.step()

        for q, targ_q, p, targ_p in zip(qf.parameters(), targ_qf.parameters(),pol.paramameters(),targ_pol.parameters()):
            targ_p.data.copy_((1 - tau) * targ_p.data + tau * p.data)
            targ_q.data.copy_((1 - tau) * targ_q.data + tau * q.data)
        qf_losses.append(qf_bellman_loss.data.cpu().numpy())
        pol_losses.append(pol_loss.data.cpu().numpy())

    logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses,
        QfLoss=qf_losses,
    )


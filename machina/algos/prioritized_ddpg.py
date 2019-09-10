"""
This is an implementation of Prioritized Experience Replay.
See https://arxiv.org/abs/1511.05952
"""

import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina.traj import traj_functional as tf
from machina import logger


def train(traj,
          pol, targ_pol, qf, targ_qf,
          optim_pol, optim_qf,
          epoch, batch_size,  # optimization hypers
          tau, gamma,
          log_enable=True,
          ):

    pol_losses = []
    qf_losses = []
    if log_enable:
        logger.log("Optimizing...")
    for batch, indices in traj.prioritized_random_batch(batch_size, epoch, return_indices=True):
        qf_bellman_loss = lf.bellman(
            qf, targ_qf, targ_pol, batch, gamma, reduction='none')
        td_loss = torch.sqrt(qf_bellman_loss*2)
        qf_bellman_loss = torch.mean(qf_bellman_loss)
        optim_qf.zero_grad()
        qf_bellman_loss.backward()
        optim_qf.step()

        pol_loss = lf.ag(pol, qf, batch)
        optim_pol.zero_grad()
        pol_loss.backward()
        optim_pol.step()

        for p, targ_p in zip(pol.parameters(), targ_pol.parameters()):
            targ_p.detach().copy_((1 - tau) * targ_p.detach() + tau * p.detach())
        for q, targ_q in zip(qf.parameters(), targ_qf.parameters()):
            targ_q.detach().copy_((1 - tau) * targ_q.detach() + tau * q.detach())

        qf_losses.append(qf_bellman_loss.detach().cpu().numpy())
        pol_losses.append(pol_loss.detach().cpu().numpy())

        traj = tf.update_pris(traj, td_loss, indices)
    if log_enable:
        logger.log("Optimization finished!")

    return {'PolLoss': pol_losses, 'QfLoss': qf_losses}

"""
This is implementation of Deep Deterministic Policy Gradient.
See https://arxiv.org/abs/1509.02971
"""

import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina import logger


def train(traj,
          pol, targ_pol, qf, targ_qf,
          optim_pol, optim_qf,
          epoch, batch_size,  # optimization hypers
          tau, gamma,  # advantage estimation
          log_enable=True,
          ):
    """
    Train function for deep deterministic policy gradient

    Parameters
    ----------
    traj : Traj
        Off policy trajectory.
    pol : Pol
        Policy.
    targ_pol : Pol
        Target Policy.
    qf : SAVfunction
        Q function.
    targ_qf : SAVfunction
        Target Q function.
    optim_pol : torch.optim.Optimizer
        Optimizer for Policy.
    optim_qf : torch.optim.Optimizer
        Optimizer for Q function.
    epoch : int
        Number of iteration.
    batch_size : int
        Number of batches.
    tau : float
        Target updating rate.
    gamma : float
        Discounting rate.
    log_enable: bool
        If True, enable logging

    Returns
    -------
    result_dict : dict
        Dictionary which contains losses information.
    """

    pol_losses = []
    qf_losses = []
    if log_enable:
        logger.log("Optimizing...")
    for batch in traj.random_batch(batch_size, epoch):
        qf_bellman_loss = lf.bellman(qf, targ_qf, targ_pol, batch, gamma)
        optim_qf.zero_grad()
        qf_bellman_loss.backward()
        optim_qf.step()

        pol_loss = lf.ag(pol, qf, batch, no_noise=True)
        optim_pol.zero_grad()
        pol_loss.backward()
        optim_pol.step()

        for p, targ_p in zip(pol.parameters(), targ_pol.parameters()):
            targ_p.detach().copy_((1 - tau) * targ_p.detach() + tau * p.detach())
        for q, targ_q in zip(qf.parameters(), targ_qf.parameters()):
            targ_q.detach().copy_((1 - tau) * targ_q.detach() + tau * q.detach())

        qf_losses.append(qf_bellman_loss.detach().cpu().numpy())
        pol_losses.append(pol_loss.detach().cpu().numpy())
    if log_enable:
        logger.log("Optimization finished!")

    return {'PolLoss': pol_losses, 'QfLoss': qf_losses}

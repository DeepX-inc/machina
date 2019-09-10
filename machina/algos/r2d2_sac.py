"""
This is an implementation of R2D2(Soft Actor Critic ver).
See https://openreview.net/pdf?id=r1lyTjAqYX and https://arxiv.org/abs/1801.01290
"""

import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina import logger
from machina.traj import traj_functional as tf


def train(traj,
          pol, qfs, targ_qfs, log_alpha,
          optim_pol, optim_qfs, optim_alpha,
          epoch, batch_size, seq_length, burn_in_length,  # optimization hypers
          tau, gamma, sampling, reparam=True,
          log_enable=True,
          ):
    """
    Train function for soft actor critic.

    Parameters
    ----------
    traj : Traj
        Off policy trajectory.
    pol : Pol
        Policy.
    qfs : list of SAVfunction
        Q function.
    targ_qfs : list of SAVfunction
        Target Q function.
    log_alpha : torch.Tensor
        Temperature parameter of entropy.
    optim_pol : torch.optim.Optimizer
        Optimizer for Policy.
    optim_qfs : list of torch.optim.Optimizer
        Optimizer for Q function.
    optim_alpha : torch.optim.Optimizer
        Optimizer for alpha.
    epoch : int
        Number of iteration.
    batch_size : int
        Number of batches.
    seq_length : int
        Length of batches.
    burn_in_length : int
        Length of batches for burn-in.
    tau : float
        Target updating rate.
    gamma : float
        Discounting rate.
    sampling : int
        Number of samping in calculating expectation.
    reparam : bool
    log_enable: bool
        If True, enable logging

    Returns
    -------
    result_dict : dict
        Dictionary which contains losses information.
    """

    pol_losses = []
    _qf_losses = []
    alpha_losses = []
    if log_enable:
        logger.log("Optimizing...")
    for batch, start_indices in traj.prioritized_random_batch_rnn(batch_size, seq_length, epoch, return_indices=True):
        batch, pol_loss, qf_losses, alpha_loss, td_losses = lf.r2d2_sac(
            pol, qfs, targ_qfs, log_alpha, batch, gamma, sampling, burn_in_length, reparam)

        optim_pol.zero_grad()
        pol_loss.backward()
        optim_pol.step()

        for optim_qf, qf_loss in zip(optim_qfs, qf_losses):
            optim_qf.zero_grad()
            qf_loss.backward()
            optim_qf.step()

        optim_alpha.zero_grad()
        alpha_loss.backward()
        optim_alpha.step()

        for qf, targ_qf in zip(qfs, targ_qfs):
            for q, targ_q in zip(qf.parameters(), targ_qf.parameters()):
                targ_q.detach().copy_((1 - tau) * targ_q.detach() + tau * q.detach())

        pol_losses.append(pol_loss.detach().cpu().numpy())
        _qf_losses.append(
            (sum(qf_losses) / len(qf_losses)).detach().cpu().numpy())
        alpha_losses.append(alpha_loss.detach().cpu().numpy())

        # update seq_pris
        train_length = seq_length - burn_in_length
        for i in range(batch_size):
            start = start_indices[i] + burn_in_length
            seq_indices = torch.arange(start, start+train_length-1)
            traj = tf.update_pris(
                traj, td_losses[:, i], seq_indices, update_epi_pris=True, seq_length=seq_length)

    if log_enable:
        logger.log("Optimization finished!")

    return dict(
        PolLoss=pol_losses,
        QfLoss=_qf_losses,
        AlphaLoss=alpha_losses
    )

"""
This is an implementation of Soft Actor Critic.
See https://arxiv.org/abs/1801.01290
"""

import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina import logger


def train(traj,
          pol, qfs, targ_qfs, log_alpha,
          optim_pol, optim_qfs, optim_alpha,
          epoch, batch_size,  # optimization hypers
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
    for batch in traj.random_batch(batch_size, epoch):
        pol_loss, qf_losses, alpha_loss = lf.sac(
            pol, qfs, targ_qfs, log_alpha, batch, gamma, sampling, reparam)

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

    if log_enable:
        logger.log("Optimization finished!")

    return dict(
        PolLoss=pol_losses,
        QfLoss=_qf_losses,
        AlphaLoss=alpha_losses
    )

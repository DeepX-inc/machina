"""
This is an implementation of Soft Actor Critic.
See https://arxiv.org/abs/1801.01290
"""

import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina import logger


def train(traj,
          pol, qf, targ_qf, log_alpha,
          optim_pol, optim_qf, optim_alpha,
          epoch, batch_size,  # optimization hypers
          tau, gamma, sampling,
          ):
    """
    Train function for soft actor critic.

    Parameters
    ----------
    traj : Traj
        Off policy trajectory.
    pol : Pol
        Policy.
    qf : SAVfunction
        Q function.
    targ_qf : SAVfunction
        Target Q function.
    log_alpha : torch.Tensor
        Temperature parameter of entropy.
    optim_pol : torch.optim.Optimizer
        Optimizer for Policy.
    optim_qf : torch.optim.Optimizer
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

    Returns
    -------
    result_dict : dict
        Dictionary which contains losses information.
    """

    qf_losses = []
    pol_losses = []
    alpha_losses = []
    logger.log("Optimizing...")
    for batch in traj.random_batch(batch_size, epoch):
        pol_loss, qf_loss, alpha_loss = lf.sac(
            pol, qf, targ_qf, log_alpha, batch, gamma, sampling)

        optim_pol.zero_grad()
        pol_loss.backward()
        optim_pol.step()

        optim_qf.zero_grad()
        qf_loss.backward()
        optim_qf.step()

        optim_alpha.zero_grad()
        alpha_loss.backward()
        optim_alpha.step()

        for q, targ_q in zip(qf.parameters(), targ_qf.parameters()):
            targ_q.detach().copy_((1 - tau) * targ_q.detach() + tau * q.detach())

        pol_losses.append(pol_loss.detach().cpu().numpy())
        qf_losses.append(qf_loss.detach().cpu().numpy())
        alpha_losses.append(alpha_loss.detach().cpu().numpy())

    logger.log("Optimization finished!")

    return dict(
        PolLoss=pol_losses,
        QfLoss=qf_losses,
        AlphaLoss=alpha_losses
    )

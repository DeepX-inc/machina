"""
This is an implementation of R2D2(Soft Actor Critic ver).
See https://openreview.net/pdf?id=r1lyTjAqYX and https://arxiv.org/abs/1801.01290
"""

import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina import logger


def train(traj,
          pol, qfs, targ_qfs, log_alpha,
          optim_pol, optim_qfs, optim_alpha,
          epoch, batch_size, seq_length, burn_in_length,  # optimization hypers
          tau, gamma, sampling, reparam=True,
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

    Returns
    -------
    result_dict : dict
        Dictionary which contains losses information.
    """

    pol_losses = []
    _qf_losses = []
    alpha_losses = []
    logger.log("Optimizing...")
    from time import time
    for batch in traj.random_batch_rnn(batch_size, seq_length, epoch):
        start = time()
        batch, pol_loss, qf_losses, alpha_loss = lf.r2d2_sac(
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
        # print(time()-start)

    logger.log("Optimization finished!")

    return dict(
        PolLoss=pol_losses,
        QfLoss=_qf_losses,
        AlphaLoss=alpha_losses
    )

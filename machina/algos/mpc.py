"""
This is an implementation of Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning.
See https://arxiv.org/abs/1708.02596
"""

import torch
import torch.nn as nn
import numpy as np

from machina import loss_functional as lf
from machina.utils import detach_tensor_dict
from machina import logger


def update_dm(dm, optim_dm, batch, target='next_obs', td=True):
    dm_loss = lf.dynamics(dm, batch, target=target, td=td)
    optim_dm.zero_grad()
    dm_loss.backward()
    optim_dm.step()

    return dm_loss.detach().cpu().numpy()


def train_dm(traj, dyn_model, optim_dm, epoch=60, batch_size=512, target='next_obs', td=True, num_epi_per_seq=1, log_enable=True):
    """
    Train function for dynamics model.

    Parameters
    ----------
    traj : Traj
        On policy trajectory.
    dyn_model : Model
        dynamics model.
    optim_dm : torch.optim.Optimizer
        Optimizer for dynamics model.
    epoch : int
        Number of iteration.
    batch_size : int
        Number of batches.
    target : str
        Target of prediction is next_obs or rews.
    td : bool
        If True, dyn_model learn temporal differance of target.
    num_epi_per_seq : int
        Number of episodes in one sequence for rnn.
    log_enable: bool
        If True, enable logging

    Returns
    -------
    result_dict : dict
        Dictionary which contains losses information.
    """

    dm_losses = []
    if log_enable:
        logger.log("Optimizing...")

    batch_size = min(batch_size, traj.num_epi)
    if dyn_model.rnn:
        iterator = traj.random_batch_rnn(
            batch_size=batch_size, epoch=epoch)
    else:
        iterator = traj.random_batch(batch_size, epoch)

    for batch in iterator:
        dm_loss = update_dm(
            dyn_model, optim_dm, batch, target=target, td=td)
        dm_losses.append(dm_loss)
    if log_enable:
        logger.log("Optimization finished!")

    return dict(DynModelLoss=dm_losses)

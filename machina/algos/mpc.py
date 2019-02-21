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


def train_dm(rl_traj, rand_traj, dyn_model, optim_dm, epoch=60, batch_size=512, rl_batch_rate=0.9, target='next_obs', td=True, num_epi_per_seq=1):
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
    rl_batch_rate : float
        rate of size of rl batch.
    target : str
        Target of prediction is next_obs or rews.
    td : bool
        If True, dyn_model learn temporal differance of target.
    num_epi_per_seq : int
        Number of episodes in one sequence for rnn.

    Returns
    -------
    result_dict : dict
        Dictionary which contains losses information.
    """

    batch_size_rl = min(int(batch_size * rl_batch_rate), rl_traj.num_epi)
    batch_size_rand = min(
        int(batch_size * (1 - rl_batch_rate)), rand_traj.num_epi)

    dm_losses = []
    logger.log("Optimizing...")

    if dyn_model.rnn:
        rl_iterator = rl_traj.iterate_rnn(
            batch_size=batch_size_rl, num_epi_per_seq=num_epi_per_seq, epoch=epoch)
        rand_iterator = rand_traj.iterate_rnn(
            batch_size=batch_size_rand, num_epi_per_seq=num_epi_per_seq, epoch=epoch)
    else:
        rl_iterator = rl_traj.iterate(batch_size, epoch)
        rand_iterator = rand_traj.iterate(batch_size, epoch)

    for rl_batch, rand_batch in zip(rl_iterator, rand_iterator):
        dyn_model.reset()
        batch = dict()
        if len(rl_batch) == 0:
            batch['obs'] = rand_batch['obs']
            batch['acs'] = rand_batch['acs']
            batch['next_obs'] = rand_batch['next_obs']
            if dyn_model.rnn:
                batch['h_masks'] = rand_batch['h_masks']
        elif len(rand_batch) == 0:
            batch['obs'] = rl_batch['obs']
            batch['acs'] = rl_batch['acs']
            batch['next_obs'] = rl_batch['next_obs']
            if dyn_model.rnn:
                batch['h_masks'] = rl_batch['h_masks']
        else:
            batch['obs'] = torch.cat(
                [rand_batch['obs'], rl_batch['obs']], dim=-2)
            batch['acs'] = torch.cat(
                [rand_batch['acs'], rl_batch['acs']], dim=-2)
            batch['next_obs'] = torch.cat(
                [rand_batch['next_obs'], rl_batch['next_obs']], dim=-2)
            if dyn_model.rnn:
                batch['h_masks'] = torch.cat(
                    [rand_batch['h_masks'], rl_batch['h_masks']], dim=1)

        dm_loss = update_dm(
            dyn_model, optim_dm, batch, target=target, td=td)
        dm_losses.append(dm_loss)
    logger.log("Optimization finished!")

    return dict(DynModelLoss=dm_losses)

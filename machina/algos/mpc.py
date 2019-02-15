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


def train_dm(rl_traj, rand_traj, dyn_model, optim_dm, epoch=60, batch_size=512, fraction_use_rl_traj=0.9, target='next_obs', td=True):
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
    Returns
    -------
    result_dict : dict
        Dictionary which contains losses information.
    """

    batch_size_rl = int(batch_size * fraction_use_rl_traj)
    batch_size_rand = batch_size - batch_size_rl

    dm_losses = []
    logger.log("Optimizing...")
    for rl_batch, rand_batch in zip(rl_traj.random_batch(batch_size_rl, epoch), rand_traj.random_batch(batch_size_rand, epoch)):
        batch = dict()

        if len(rl_batch) == 0:
            batch['obs'] = rand_batch['obs']
            batch['acs'] = rand_batch['acs']
            batch['next_obs'] = rand_batch['next_obs']
        else:
            batch['obs'] = rl_batch['obs']
            batch['acs'] = rl_batch['acs']
            batch['next_obs'] = rl_batch['next_obs']

        if batch_size_rand > 0:
            batch['obs'] = torch.cat([batch['obs'], rand_batch['obs']], dim=0)
            batch['acs'] = torch.cat([batch['acs'], rand_batch['acs']], dim=0)
            batch['next_obs'] = torch.cat(
                [batch['next_obs'], rand_batch['next_obs']], dim=0)

        dm_loss = update_dm(
            dyn_model, optim_dm, batch, target=target, td=td)
        dm_losses.append(dm_loss)
    logger.log("Optimization finished!")

    return dict(DynModelLoss=dm_losses)

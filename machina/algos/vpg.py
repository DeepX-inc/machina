"""
This is an implementation of Vanilla Policy Gradient.
"""

import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina import logger


def update_pol(pol, optim_pol, batch):
    pol_loss = lf.pg(pol, batch)
    optim_pol.zero_grad()
    pol_loss.backward()
    optim_pol.step()
    return pol_loss.detach().cpu().numpy()


def update_vf(vf, optim_vf, batch):
    vf_loss = lf.monte_carlo(vf, batch)
    optim_vf.zero_grad()
    vf_loss.backward()
    optim_vf.step()
    return vf_loss.detach().cpu().numpy()


def train(traj, pol, vf,
          optim_pol, optim_vf,
          epoch, batch_size,  # optimization hypers
          large_batch,
          log_enable=True,
          ):
    """
    Train function for vanila policy gradient.

    Parameters
    ----------
    traj : Traj
        On policy trajectory.
    pol : Pol
        Policy.
    vf : SVfunction
        V function.
    optim_pol : torch.optim.Optimizer
        Optimizer for Policy.
    optim_vf : torch.optim.Optimizer
        Optimizer for V function.
    epoch : int
        Number of iteration.
    batch_size : int
        Number of batches.
    larget_batch : bool
        If True, batch is provided as whole trajectory.
    log_enable: bool
        If True, enable logging

    Returns
    -------
    result_dict : dict
        Dictionary which contains losses information.
    """
    pol_losses = []
    vf_losses = []
    if log_enable:
        logger.log("Optimizing...")
    if large_batch:
        for batch in traj.full_batch(epoch):
            pol_loss = update_pol(pol, optim_pol, batch)
            vf_loss = update_vf(vf, optim_vf, batch)

            pol_losses.append(pol_loss)
            vf_losses.append(vf_loss)
    else:
        for batch in traj.iterate(batch_size, epoch):
            pol_loss = update_pol(pol, optim_pol, batch)
            vf_loss = update_vf(vf, optim_vf, batch)

            pol_losses.append(pol_loss)
            vf_losses.append(vf_loss)
    if log_enable:
        logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses, VfLoss=vf_losses)

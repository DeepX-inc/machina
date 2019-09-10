"""
Implementation of On-policy and Teacher distillation technique
"""

import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina import logger


def update_pol(student_pol, teacher_pol, optim_pol, batch):
    """
    Update function of Student-policy

    Parameters
    ----------
    student_pol : Pol
        Student Policy.
    teacher_pol : Pol
        Teacher Policy
    optim_pol : torch.optim.Optimizer
        Optimizer for Policy
    batch : dict
        Batch of trajectory

    Returns
    -------
    pol_loss : ndarray
        Loss of student policy
    """

    pol_loss = lf.shannon_cross_entropy(student_pol, teacher_pol, batch)
    optim_pol.zero_grad()
    pol_loss.backward()
    optim_pol.step()
    return pol_loss.detach().cpu().numpy()


def train(traj, student_pol, teacher_pol, student_optim, epoch, batchsize, num_epi_per_seq=1, log_enable=True):
    s_pol_losses = []
    if log_enable:
        logger.log("Optimizing...")
    iterator = traj.iterate(batchsize, epoch) if not student_pol.rnn else traj.iterate_rnn(
        batchsize=batchsize, num_epi_per_seq=num_epi_per_seq, epoch=epoch)
    for batch in iterator:
        s_pol_loss = update_pol(
            student_pol=student_pol, teacher_pol=teacher_pol, optim_pol=student_optim, batch=batch)
        s_pol_losses.append(s_pol_loss)

    if log_enable:
        logger.log('Optimization finished')
    return dict(S_Pol_loss=s_pol_losses)

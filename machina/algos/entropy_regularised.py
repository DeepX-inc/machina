"""
Implementation of Entropy_regularised technique technique
"""

import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina import logger
from collections import OrderedDict

def update_pol(student_pol, batch):
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

    total_rew = lf.entropy_regularized_rew(student_pol, batch)
    grads = torch.autograd.grad(total_rew, student_pol.parameters())
    updated_params = OrderedDict()
    for (name, param), grad in zip(student_pol.named_parameters(), grads):
        updated_params[name] = param + grad
    student_pol.load_state_dict(updated_params)
    return total_rew.detach().cpu().numpy()


def train(traj, student_pol, teacher_pol, student_optim, epoch, batchsize, num_epi_per_seq=1):
    s_pol_losses = []
    logger.log("Optimizing...")
    iterator = traj.iterate(batchsize, epoch) if not student_pol.rnn else traj.iterate_rnn(
        batchsize=batchsize, num_epi_per_seq=num_epi_per_seq, epoch=epoch)
    for batch in iterator:
        s_pol_loss = update_pol(
            student_pol=student_pol, teacher_pol=teacher_pol, optim_pol=student_optim, batch=batch)
        s_pol_losses.append(s_pol_loss)

    logger.log('Optimization finished')
    return dict(S_Pol_loss=s_pol_losses)

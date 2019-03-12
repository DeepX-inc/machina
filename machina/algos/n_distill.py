"""
Implementation of On-policy and Teacher distillation technique
"""

import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina import logger

from collections import OrderedDict

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
    updated_params : Ordered dict
        Dictionary containing updated parameters
    """

    #First there is the loss-term
    pol_loss = lf.shannon_cross_entropy(student_pol=student_pol, teacher_pol=teacher_pol, batch=batch)

    # Reward term is the llh times cross entropy of the next observations
    reward_term = lf.shannon_cross_entropy_next_obs(student_pol=student_pol, teacher_pol=teacher_pol, batch=batch)
    llh = lf.log_likelihood(student_pol, batch)
    tot_reward = llh*reward_term

    # calculate gradients
    pol_loss_grad  = torch.autograd.grad(pol_loss, student_pol.net.parameters())
    pol_reward_grad  = torch.autograd.grad(tot_reward, student_pol.net.parameters())

    tot_grads = pol_loss_grad+pol_reward_grad
    updated_params = OrderedDict()
    for (name, param), grad in zip(student_pol.net.named_parameters(), tot_grads):
        updated_params[name] = param - grad

    return [pol_loss.detach().cpu().numpy(), updated_params]


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

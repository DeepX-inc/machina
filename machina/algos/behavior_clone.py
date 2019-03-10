"""
This is an implementation of Behavioral Cloning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from machina import loss_functional as lf


def update_pol(pol, optim_pol, batch):
    pol_loss = lf.log_likelihood(pol, batch)
    optim_pol.zero_grad()
    pol_loss.backward()
    optim_pol.step()
    return pol_loss.detach().cpu().numpy()


def train(expert_traj, pol, optim_pol, batch_size):
    pol_losses = []
    iterater = expert_traj.iterate_once(batch_size)
    for batch in iterater:
        pol_loss = update_pol(pol, optim_pol, batch)
        pol_losses.append(pol_loss)
    return dict(PolLoss=pol_losses)


def test(expert_traj, pol):
    pol.eval()
    iterater = expert_traj.full_batch(epoch=1)
    for batch in iterater:
        with torch.no_grad():
            pol_loss = lf.log_likelihood(pol, batch)
    return dict(TestPolLoss=[float(pol_loss.detach().cpu().numpy())])

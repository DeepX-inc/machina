import torch
import torch.nn as nn
from machina.utils import Variable, torch2torch, np2torch
from machina.misc import logger
import numpy as np
import os

def make_pol_loss(pol, batch):
    obs = Variable(batch['obs'].float())
    _, _, param = pol(obs)
    pol_loss =torch.mean(0.5 *( Variable(batch['acs'].float()) - param['mean'])**2)
    return pol_loss

def train(expert_data,pol,optim_pol,
        epoch_pol, batch_size
        ):
    pol_losses = []
    for iteration, batch in expert_data.iterate(batch_size):
        pol_loss = make_pol_loss(pol, batch)
        optim_pol.zero_grad()
        pol_loss.backward()
        optim_pol.step()
        pol_losses.append(pol_loss.data.cpu().numpy())
    validation_loss = make_pol_loss(pol, expert_data.test_data).data.cpu().numpy()

    return {'PolLoss': pol_losses, 'ValidationPolLoss' : [validation_loss]}
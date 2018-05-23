import torch
import torch.nn as nn
from machina.utils import Variable, torch2torch, np2torch
from machina.misc import logger
import numpy as np
import os

def make_pol_loss(pol, transition_model, batch):
    obs = Variable(batch['obs'].float())
    _, _, param = pol(obs)
    pol_loss =torch.mean(0.5 *( Variable(batch['acs'].float()) - param['mean'])**2)
    return pol_loss

def train_only_pol(expert_data,
        pol, transition_model,
        optim_pol,
        epoch_pol, batch_size
        ):
    pol_losses = []
    logger.log("Optimizing...")
    for batch in expert_data.iterate(batch_size, epoch_pol):
        pol_loss = make_pol_loss(pol, transition_model, batch)
        optim_pol.zero_grad()
        pol_loss.backward()
        optim_pol.step()
        pol_losses.append(pol_loss.data.cpu().numpy())
    logger.log("Optimization finished!")
    return {'PolLoss': pol_losses}
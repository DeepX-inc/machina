import torch
import torch.nn as nn
from machina.utils import Variable
from machina.misc import logger

def make_pol_loss(pol, batch):
    obs = Variable(batch['obs'])
    acs = Variable(batch['acs'])
    advs = Variable(batch['advs'])
    _, _, pd_params = pol(obs)
    llh = pol.pd.llh(acs, pd_params)

    pol_loss = - torch.mean(llh * advs)
    return pol_loss

def update_pol(pol, optim_pol, batch):
    pol_loss = make_pol_loss(pol, batch)
    optim_pol.zero_grad()
    pol_loss.backward()
    optim_pol.step()
    return pol_loss.data.cpu().numpy()

def make_vf_loss(vf, batch):
    obs = Variable(batch['obs'])
    rets = Variable(batch['rets'])
    vf_loss = 0.5 * torch.mean((vf(obs) - rets)**2)
    return vf_loss

def update_vf(vf, optim_vf, batch):
    vf_loss = make_vf_loss(vf, batch)
    optim_vf.zero_grad()
    vf_loss.backward()
    optim_vf.step()
    return vf_loss.data.cpu().numpy()

def train(data, pol, vf,
        optim_pol, optim_vf,
        epoch, batch_size,# optimization hypers
        gamma, lam, # advantage estimation
        large_batch,
        ):

    pol_losses = []
    vf_losses = []
    logger.log("Optimizing...")
    if large_batch:
        for batch in data.full_batch(epoch):
            pol_loss = update_pol(pol, optim_pol, batch)
            vf_loss = update_vf(vf, optim_vf, batch)

            pol_losses.append(pol_loss)
            vf_losses.append(vf_loss)
    else:
        for batch in data.iterate(batch_size, epoch):
            pol_loss = update_pol(pol, optim_pol, batch)
            vf_loss = update_vf(vf, optim_vf, batch)

            pol_losses.append(pol_loss)
            vf_losses.append(vf_loss)
    logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses, VfLoss=vf_losses)

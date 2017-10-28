import torch
import torch.nn as nn
from torch.autograd import Variable
from ..misc import logger

def make_pol_loss(pol, batch, clip_param):
    obs = Variable(torch.from_numpy(batch['obs']).float())
    acs = Variable(torch.from_numpy(batch['acs']).float())
    advs = Variable(torch.from_numpy(batch['advs']).float())
    old_llh = Variable(pol.pd.llh(
        torch.from_numpy(batch['acs']).float(),
        torch.from_numpy(batch['mean']).float(),
        torch.from_numpy(batch['log_std']).float()
    ))
    _, _, pd_params = pol(obs)
    new_llh = pol.pd.llh(acs, pd_params['mean'], pd_params['log_std'])
    ratio = torch.exp(new_llh - old_llh)
    pol_loss1 = ratio * advs
    pol_loss2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advs
    pol_loss =  -torch.mean(torch.min(pol_loss1, pol_loss2))
    return pol_loss

def update_pol(pol, optim_pol, batch, clip_param):
    pol_loss = make_pol_loss(pol, batch, clip_param)
    optim_pol.zero_grad()
    pol_loss.backward()
    optim_pol.step()
    return pol_loss.data.cpu().numpy()

def make_vf_loss(vf, batch, clip_param):
    obs = Variable(torch.from_numpy(batch['obs']).float())
    rets = Variable(torch.from_numpy(batch['rets']).float())
    vs = Variable(torch.from_numpy(batch['vs']).float())

    vfloss1 = (vf(obs) - rets)**2
    vpredclipped = vs + torch.clamp(vf(obs) - vs, -clip_param, clip_param)
    vfloss2 = (vpredclipped - rets)**2
    vf_loss = 0.5 * torch.mean(torch.max(vfloss1, vfloss2))
    return vf_loss

def update_vf(vf, optim_vf, batch, clip_param):
    vf_loss = make_vf_loss(vf, batch, clip_param)
    optim_vf.zero_grad()
    vf_loss.backward()
    optim_vf.step()
    return vf_loss.data.cpu().numpy()

def train(data, pol, vf,
        clip_param,
        optim_pol, optim_vf,
        epoch, batch_size,# optimization hypers
        gamma, lam, # advantage estimation
        ):

    pol_losses = []
    vf_losses = []
    logger.log("Optimizing...")
    for batch in data.iterate(batch_size, epoch):
        pol_loss = update_pol(pol, optim_pol, batch, clip_param)
        vf_loss = update_vf(vf, optim_vf, batch, clip_param)

        pol_losses.append(pol_loss)
        vf_losses.append(vf_loss)
    logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses, VfLoss=vf_losses)

import torch
import torch.nn as nn
from ..utils import Variable
from ..misc import logger

def make_pol_loss(pol, batch, kl_beta):
    obs = Variable(torch.from_numpy(batch['obs']).float())
    acs = Variable(torch.from_numpy(batch['acs']).float())
    advs = Variable(torch.from_numpy(batch['advs']).float())

    old_mean = Variable(torch.from_numpy(batch['mean']).float())
    old_log_std = Variable(torch.from_numpy(batch['log_std']).float())

    old_llh = Variable(pol.pd.llh(
        torch.from_numpy(batch['acs']).float(),
        torch.from_numpy(batch['mean']).float(),
        torch.from_numpy(batch['log_std']).float()
    ))

    _, _, pd_params = pol(obs)
    new_llh = pol.pd.llh(acs, pd_params['mean'], pd_params['log_std'])
    ratio = torch.exp(new_llh - old_llh)
    pol_loss = ratio * advs

    kl = pol.pd.kl_pq(p_mean=old_mean, p_log_std=old_log_std,
            q_mean=pd_params['mean'], q_log_std=pd_params['log_std'])

    pol_loss -= kl_beta * kl
    pol_loss = - torch.mean(pol_loss)

    return pol_loss

def update_pol(pol, optim_pol, batch, kl_beta):
    pol_loss = make_pol_loss(pol, batch, kl_beta)
    optim_pol.zero_grad()
    pol_loss.backward()
    optim_pol.step()
    return pol_loss.data.cpu().numpy()

def make_vf_loss(vf, batch):
    obs = Variable(torch.from_numpy(batch['obs']).float())
    rets = Variable(torch.from_numpy(batch['rets']).float())
    vf_loss = 0.5 * torch.mean((vf(obs) - rets)**2)
    return vf_loss

def update_vf(vf, optim_vf, batch):
    vf_loss = make_vf_loss(vf, batch)
    optim_vf.zero_grad()
    vf_loss.backward()
    optim_vf.step()
    return vf_loss.data.cpu().numpy()

def train(data, pol, vf,
        kl_beta, kl_targ,
        optim_pol, optim_vf,
        epoch, batch_size,# optimization hypers
        gamma, lam, # advantage estimation
        ):

    pol_losses = []
    vf_losses = []
    logger.log("Optimizing...")
    for batch in data.iterate(batch_size, epoch):
        pol_loss = update_pol(pol, optim_pol, batch, kl_beta)
        vf_loss = update_vf(vf, optim_vf, batch)

        pol_losses.append(pol_loss)
        vf_losses.append(vf_loss)

    batch = next(data.full_batch())
    _, _, pd_params = pol(Variable(torch.from_numpy(batch['obs']).float()))
    kl_mean = torch.mean(
        pol.pd.kl_pq(
            torch.from_numpy(batch['mean']),
            torch.from_numpy(batch['log_std']),
            pd_params['mean'].data,
            pd_params['log_std'].data
        )
    )
    if kl_mean > 1.3 * kl_targ:
        new_kl_beta = 1.5 * kl_beta
    elif kl_mean < 0.7 * kl_targ:
        new_kl_beta = kl_beta / 1.5
    else:
        new_kl_beta = kl_beta
    logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses, VfLoss=vf_losses, new_kl_beta=new_kl_beta, kl_mean=kl_mean)


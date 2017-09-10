import torch
import torch.nn as nn
from torch.autograd import Variable
from ..misc import logger

def train(data, pol, vfunc,
        clip_param,
        mc_optim, # machina's optimizer
        optim_epochs, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        ):

    data.set_adv(vfunc, gamma, lam, centerize=True)
    pol_surrs = []
    vf_losses = []
    logger.log("Optimizing...")
    for _ in range(optim_epochs):
        for batch in data.iterate_once(optim_batchsize):
            ob = Variable(torch.from_numpy(batch['obs']).float())
            ac = Variable(torch.from_numpy(batch['acs']).float())
            adv = Variable(torch.from_numpy(batch['advs']).float())
            ret = Variable(torch.from_numpy(batch['rets']).float())
            v = Variable(torch.from_numpy(batch['vs']).float())
            old_llh = Variable(pol.pd.llh(
                torch.from_numpy(batch['acs']).float(),
                torch.from_numpy(batch['mean']).float(),
                torch.from_numpy(batch['log_std']).float()
            ))
            _, _, pd_params = pol(ob)
            new_llh = pol.pd.llh(ac, pd_params['mean'], pd_params['log_std'])
            ratio = torch.exp(new_llh - old_llh)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
            pol_surr =  -torch.mean(torch.min(surr1, surr2))
            vfloss1 = (vfunc(ob) - ret)**2
            vpredclipped = v + torch.clamp(vfunc(ob) - v, -clip_param, clip_param)
            vfloss2 = (vpredclipped - ret)**2
            vf_loss = 0.5 * torch.mean(torch.max(vfloss1, vfloss2))
            total_loss = pol_surr + vf_loss
            mc_optim.zero_grad()
            total_loss.backward()
            mc_optim.step()
            pol_surrs.append(pol_surr.data.numpy())
            vf_losses.append(vf_loss.data.numpy())
    logger.log("Optimization finished!")

    logger.record_tabular_misc_stat('PolLoss', pol_surrs)
    logger.record_tabular_misc_stat('VLoss', vf_losses)
    logger.record_tabular('Episode', data.num_epi)

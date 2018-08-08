# Copyright 2018 DeepX Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
from machina.misc import logger

def make_pol_loss(pol, batch, kl_beta):
    obs = batch['obs']
    acs = batch['acs']
    advs = batch['advs']

    old_mean = batch['mean']
    old_log_std = batch['log_std']

    old_llh = pol.pd.llh(
        batch['acs'],
        batch
    )

    _, _, pd_params = pol(obs)
    new_llh = pol.pd.llh(acs, pd_params)
    ratio = torch.exp(new_llh - old_llh)
    pol_loss = ratio * advs

    kl = pol.pd.kl_pq(
        batch,
        pd_params
    )

    pol_loss -= kl_beta * kl
    pol_loss = - torch.mean(pol_loss)

    return pol_loss

def update_pol(pol, optim_pol, batch, kl_beta):
    pol_loss = make_pol_loss(pol, batch, kl_beta)
    optim_pol.zero_grad()
    pol_loss.backward()
    optim_pol.step()
    return pol_loss.detach().numpy()

def make_vf_loss(vf, batch):
    obs = batch['obs']
    rets = batch['rets']
    vf_loss = 0.5 * torch.mean((vf(obs) - rets)**2)
    return vf_loss

def update_vf(vf, optim_vf, batch):
    vf_loss = make_vf_loss(vf, batch)
    optim_vf.zero_grad()
    vf_loss.backward()
    optim_vf.step()
    return vf_loss.detach().numpy()

def train(data, pol, vf,
        kl_beta, kl_targ,
        optim_pol, optim_vf,
        epoch, batch_size,# optimization hypers
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
    with torch.no_grad():
        _, _, pd_params = pol(batch['obs'])
        kl_mean = torch.mean(
            pol.pd.kl_pq(
                batch,
                pd_params
            )
        ).item()
    if kl_mean > 1.3 * kl_targ:
        new_kl_beta = 1.5 * kl_beta
    elif kl_mean < 0.7 * kl_targ:
        new_kl_beta = kl_beta / 1.5
    else:
        new_kl_beta = kl_beta
    logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses, VfLoss=vf_losses, new_kl_beta=new_kl_beta, kl_mean=kl_mean)


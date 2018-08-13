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
#
# This is an implementation of Soft Actor Critic.
# See https://arxiv.org/abs/1801.01290
#

import torch
import torch.nn as nn
from machina.misc import logger

def make_pol_loss(pol, qf, vf, batch, sampling):
    obs = batch['obs']

    pol_loss = 0
    _, _, pd_params = pol(obs)
    # TODO: fast sampling
    for _ in range(sampling):
        acs = pol.pd.sample(pd_params)
        llh = pol.pd.llh(acs.detach(), pd_params)
        pol_loss += llh * (llh.detach() - qf(obs, acs).detach() + vf(obs).detach())
    pol_loss /= sampling

    return torch.mean(pol_loss)

def make_qf_loss(qf, vf, batch, gamma):
    obs = batch['obs']
    acs = batch['acs']
    rews = batch['rews']
    next_obs = batch['next_obs']
    terminals = batch['terminals']

    targ = rews + gamma * vf(next_obs) * (1 - terminals)
    targ = targ.detach()

    return 0.5 * torch.mean((qf(obs, acs) - targ)**2)

def make_vf_loss(pol, qf, vf, batch, sampling):
    obs = batch['obs']

    targ = 0
    _, _, pd_params = pol(obs)
    # TODO: fast sampling
    for _ in range(sampling):
        acs = pol.pd.sample(pd_params)
        llh = pol.pd.llh(acs, pd_params)
        targ += qf(obs, acs) - llh
    targ /= sampling
    targ = targ.detach()

    return 0.5 * torch.mean((vf(obs) - targ)**2)

def train(off_data,
        pol, qf, vf,
        optim_pol, optim_qf, optim_vf,
        epoch, batch_size,# optimization hypers
        gamma, sampling,
        ):

    vf_losses = []
    qf_losses = []
    pol_losses = []
    logger.log("Optimizing...")
    for batch in off_data.iterate(batch_size, epoch):
        vf_loss = make_vf_loss(pol, qf, vf, batch, sampling)
        optim_vf.zero_grad()
        vf_loss.backward()
        optim_vf.step()

        qf_loss = make_qf_loss(qf, vf, batch, gamma)
        optim_qf.zero_grad()
        qf_loss.backward()
        optim_qf.step()

        pol_loss = make_pol_loss(pol, qf, vf, batch, sampling)
        optim_pol.zero_grad()
        pol_loss.backward()
        optim_pol.step()

        vf_losses.append(vf_loss.detach().cpu().numpy())
        qf_losses.append(qf_loss.detach().cpu().numpy())
        pol_losses.append(pol_loss.detach().cpu().numpy())

    logger.log("Optimization finished!")

    return dict(
        VfLoss=vf_losses,
        QfLoss=qf_losses,
        PolLoss=pol_losses,
    )


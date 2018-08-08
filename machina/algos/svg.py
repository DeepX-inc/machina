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

def make_pol_loss(pol, qf, batch, sampling, kl_coeff=0):
    obs = batch['obs']

    q = 0
    _, _, pd_params = pol(obs)
    # TODO: fast sampling
    for _ in range(sampling):
        acs = pol.pd.sample(pd_params)
        q += qf(obs, acs)
    q /= sampling

    pol_loss = - torch.mean(q)

    _, _, pd_params = pol(obs)
    kl = pol.pd.kl_pq(
        {k:d.detach() for k, d in pd_params.items()},
        pd_params
    )
    mean_kl = torch.mean(kl)

    return pol_loss + mean_kl * kl_coeff

def update_pol(pol, qf, optim_pol, batch, sampling):
    pol_loss = make_pol_loss(pol, qf, batch, sampling)
    optim_pol.zero_grad()
    pol_loss.backward()
    optim_pol.step()
    return pol_loss.detach().numpy()

def make_bellman_loss(qf, targ_qf, pol, batch, gamma, sampling):
    obs = batch['obs']
    acs = batch['acs']
    rews = batch['rews']
    next_obs = batch['next_obs']
    terminals = batch['terminals']
    expected_next_q = 0
    _, _, pd_params = pol(next_obs)
    next_means, next_log_stds = pd_params['mean'], pd_params['log_std']
    for _ in range(sampling):
        next_acs = next_means + torch.randn_like(next_means) * torch.exp(next_log_stds)
        expected_next_q += targ_qf(next_obs, next_acs)
    expected_next_q /= sampling
    targ = rews + gamma * expected_next_q * (1 - terminals)
    targ = targ.detach()

    return 0.5 * torch.mean((qf(obs, acs) - targ)**2)

def make_mc_loss(qf, batch):
    obs = batch['obs']
    acs = batch['acs']
    rets = batch['rets']
    return 0.5 * torch.mean((qf(obs, acs) - rets)**2)

def train(off_data,
        pol, qf, targ_qf,
        optim_pol, optim_qf,
        epoch, batch_size,# optimization hypers
        tau, gamma, # advantage estimation
        sampling,
        ):

    pol_losses = []
    qf_losses = []
    logger.log("Optimizing...")
    for batch in off_data.iterate(batch_size, epoch):
        qf_bellman_loss = make_bellman_loss(qf, targ_qf, pol, batch, gamma, sampling)
        optim_qf.zero_grad()
        qf_bellman_loss.backward()
        optim_qf.step()

        pol_loss = make_pol_loss(pol, qf, batch, sampling)
        optim_pol.zero_grad()
        pol_loss.backward()
        optim_pol.step()

        for p, targ_p in zip(qf.parameters(), targ_qf.parameters()):
            targ_p.copy_((1 - tau) * targ_p + tau * p)
        qf_losses.append(qf_bellman_loss.detach().jnumpy())
        pol_losses.append(pol_loss.detach().numpy())

    logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses,
        QfLoss=qf_losses,
    )


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
import torch.nn.functional as F
from machina.misc import logger
from machina.algos import trpo, ppo_kl, ppo_clip


def make_discrim_expert_loss(discrim, batch):
    obs = batch['obs']
    acs = batch['acs']
    len = obs.shape[0]
    pred = discrim(obs, acs)
    discrim_expert_loss = F.binary_cross_entropy(pred, torch.ones(len))
    return discrim_expert_loss


def make_discrim_agent_loss(discrim, batch):
    obs = batch['obs']
    acs = batch['acs']
    len = obs.shape[0]
    pred = discrim(obs, acs)
    discrim_agent_loss = F.binary_cross_entropy(pred, torch.zeros(len))
    return discrim_agent_loss


def make_discrim_loss(discrim, batch, expert_batch):
    discrim_expert_loss = make_discrim_expert_loss(discrim, expert_batch)
    discrim_agent_loss = make_discrim_agent_loss(discrim, batch)
    return discrim_expert_loss + discrim_agent_loss


def update_discrim_loss(discrim, optim_discrim, batch, expert_batch, ent_beta=0.001):
    discrim_loss = make_discrim_loss(discrim, batch, expert_batch)
    optim_discrim.zero_grad()
    discrim_loss.backward()
    optim_discrim.step()
    return discrim_loss.detach().cpu().numpy()


def train(data, expert_data, pol, vf, discrim,
          optim_vf, optim_discim,
          pol_epoch=1, vf_epoch=5, discrim_epoch=1, vf_batch_size=128,
          pol_step=3, vf_step=3, discrim_step=1,  # optimization hypers
          max_kl=0.01, num_cg=10, damping=0.1, pol_ent_beta=0, discrim_ent_beta=0
          ):
    pol_losses = []
    vf_losses = []
    discrim_losses = []

    logger.log("Optimizing...")
    for batch in data.full_batch(step=pol_step, epoch=pol_epoch):
        pol_loss = trpo.update_pol(pol, batch, max_kl=max_kl, num_cg=num_cg, damping=damping, ent_beta=pol_ent_beta)
        pol_losses.append(pol_loss)
        if 'Normalized' in vf.__class__.__name__:
            vf.set_mean(torch.mean(batch['rets'], 0, keepdim=True))
            vf.set_std(torch.std(batch['rets'], 0, keepdim=True))

    for batch in data.iterate(vf_batch_size, step=vf_step, epoch=vf_epoch):
        vf_loss = trpo.update_vf(vf, optim_vf, batch)
        vf_losses.append(vf_loss)

    for agent_batch, expert_batch in zip(data.full_batch(step=discrim_step, epoch=discrim_epoch),
                                         expert_data.iterate(batch_size=data.n_of_data_map[data._next_data_map_id],
                                                             epoch=discrim_step * discrim_epoch)):
        discrim_loss = update_discrim_loss(discrim, optim_discim, agent_batch, expert_batch, ent_beta=discrim_ent_beta)
        discrim_losses.append(discrim_loss)
    logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses, VfLoss=vf_losses, DiscrimLoss=discrim_losses)
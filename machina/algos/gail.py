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
# This is an implementation of Generative Adversarial Imiation Learning
# See https://arxiv.org/abs/1606.03476
#


import torch
import torch.nn as nn
import torch.nn.functional as F

from machina import  loss_functional as lf
from machina import logger
from machina.algos import trpo, ppo_kl, ppo_clip
from machina.utils import get_device


def update_discrim(discrim, optim_discrim, agent_batch, expert_batch, ent_beta=0.001):
    discrim_loss = lf.cross_ent(discrim, agent_batch, expert_or_agent=0, ent_beta=ent_beta)
    discrim_loss += lf.cross_ent(discrim, expert_batch, expert_or_agent=1, ent_beta=ent_beta)
    discrim_loss /= 2
    optim_discrim.zero_grad()
    discrim_loss.backward()
    optim_discrim.step()
    return discrim_loss.detach().cpu().numpy()

def train(agent_traj, expert_traj, pol, vf, discrim,
        optim_vf, optim_discim,
        rl_type='trpo',
        pol_ent_beta=0, discrim_ent_beta=0,
        epoch=1,
        batch_size=64, discrim_batch_size=32,
        num_epi_per_seq=1, discrim_step=1, # optimization hypers
        damping=0.1, max_kl=0.01, num_cg=10, #trpo hypers
        optim_pol=None,
        clip_param=0.2, max_grad_norm=0.5, clip_vfunc=False, kl_beta = 1, kl_targ = 0.01 #ppo hypers
        ):

    pol_losses = []
    vf_losses = []
    discrim_losses = []
    
    logger.log("Optimizing...")
    if rl_type == 'trpo':
        iterator = agent_traj.full_batch(1) if not pol.rnn else agent_traj.iterate_rnn(batch_size=agent_traj.num_epi)
        for batch in iterator:
            pol_loss = trpo.update_pol(pol, batch, max_kl=max_kl, num_cg=num_cg, damping=damping, ent_beta=pol_ent_beta)
            pol_losses.append(pol_loss)

        iterator = agent_traj.iterate(batch_size, epoch) if not pol.rnn else agent_traj.iterate_rnn(batch_size=batch_size,
                                                                                        num_epi_per_seq=num_epi_per_seq,
                                                                                        epoch=epoch)
        for batch in iterator:
            vf_loss = trpo.update_vf(vf, optim_vf, batch)
            vf_losses.append(vf_loss)
        new_kl_beta = 0
        kl_mean = 0
    elif rl_type == 'ppo_clip':
        iterator = agent_traj.iterate(batch_size, epoch) if not pol.rnn else agent_traj.iterate_rnn(batch_size=batch_size,
                                                                                        num_epi_per_seq=num_epi_per_seq,
                                                                                        epoch=epoch)
        for batch in iterator:
            pol_loss = ppo_clip.update_pol(pol, optim_pol, batch, clip_param, pol_ent_beta, max_grad_norm)
            vf_loss = ppo_clip.update_vf(vf, optim_vf, batch, clip_param, clip_vfunc, max_grad_norm)

            pol_losses.append(pol_loss)
            vf_losses.append(vf_loss)
        new_kl_beta = 0
        kl_mean = 0
    else:
        iterator = agent_traj.iterate(batch_size, epoch) if not pol.rnn else agent_traj.iterate_rnn(batch_size=batch_size,
                                                                                        num_epi_per_seq=num_epi_per_seq,
                                                                                        epoch=epoch)
        for batch in iterator:
            pol_loss = ppo_kl.update_pol(pol, optim_pol, batch, kl_beta, max_grad_norm, pol_ent_beta)
            vf_loss = ppo_kl.update_vf(vf, optim_vf, batch)

            pol_losses.append(pol_loss)
            vf_losses.append(vf_loss)

        iterator = agent_traj.full_batch(1) if not pol.rnn else agent_traj.iterate_rnn(batch_size=agent_traj.num_epi)
        batch = next(iterator)
        with torch.no_grad():
            pol.reset()
            if pol.rnn:
                _, _, pd_params = pol(batch['obs'], h_masks=batch['h_masks'])
            else:
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

    agent_iterator = agent_traj.step(batch_size=discrim_batch_size, step=discrim_step)
    expert_iterator = expert_traj.step(batch_size=discrim_batch_size, step=discrim_step)
    for agent_batch, expert_batch in zip(agent_iterator, expert_iterator):
        discrim_loss = update_discrim(discrim, optim_discim, agent_batch, expert_batch, ent_beta=discrim_ent_beta)
        discrim_losses.append(discrim_loss)
    logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses, VfLoss=vf_losses, DiscrimLoss=discrim_losses, new_kl_beta=new_kl_beta, kl_mean=kl_mean)
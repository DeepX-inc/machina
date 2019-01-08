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
# This is an implementation of Proximal Policy Optimization
# in which gradient is clipped by the size especially.
# See https://arxiv.org/abs/1707.06347
#


import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina import logger


def update_pol(pol, optim_pol, batch, clip_param, ent_beta, max_grad_norm):
    pol_loss = lf.pg_clip(pol, batch, clip_param, ent_beta)
    optim_pol.zero_grad()
    pol_loss.backward()
    torch.nn.utils.clip_grad_norm_(pol.parameters(), max_grad_norm)
    optim_pol.step()
    return pol_loss.detach().cpu().numpy()


def update_vf(vf, optim_vf, batch, clip_param, clip, max_grad_norm):
    vf_loss = lf.monte_carlo(vf, batch, clip_param, clip)
    optim_vf.zero_grad()
    vf_loss.backward()
    torch.nn.utils.clip_grad_norm_(vf.parameters(), max_grad_norm)
    optim_vf.step()
    return vf_loss.detach().cpu().numpy()


def train(traj, pol, vf,
          optim_pol, optim_vf,
          epoch, batch_size, num_epi_per_seq=1,  # optimization hypers
          clip_param=0.2, ent_beta=1e-3,
          max_grad_norm=0.5,
          clip_vfunc=False
          ):

    pol_losses = []
    vf_losses = []
    logger.log("Optimizing...")
    iterator = traj.iterate(batch_size, epoch) if not pol.rnn else traj.iterate_rnn(
        batch_size=batch_size, num_epi_per_seq=num_epi_per_seq, epoch=epoch)
    for batch in iterator:
        pol_loss = update_pol(pol, optim_pol, batch,
                              clip_param, ent_beta, max_grad_norm)
        vf_loss = update_vf(vf, optim_vf, batch, clip_param,
                            clip_vfunc, max_grad_norm)

        pol_losses.append(pol_loss)
        vf_losses.append(vf_loss)
    logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses, VfLoss=vf_losses)

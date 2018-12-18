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
# This is an implementation of Vanilla Policy Gradient.


import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina.misc import logger


def update_pol(pol, optim_pol, batch):
    pol_loss = lf.pg(pol, batch)
    optim_pol.zero_grad()
    pol_loss.backward()
    optim_pol.step()
    return pol_loss.detach().cpu().numpy()

def update_vf(vf, optim_vf, batch):
    vf_loss = lf.monte_carlo(vf, batch)
    optim_vf.zero_grad()
    vf_loss.backward()
    optim_vf.step()
    return vf_loss.detach().cpu().numpy()

def train(traj, pol, vf,
        optim_pol, optim_vf,
        epoch, batch_size,# optimization hypers
        gamma, lam, # advantage estimation
        large_batch,
        ):

    pol_losses = []
    vf_losses = []
    logger.log("Optimizing...")
    if large_batch:
        for batch in traj.full_batch(epoch):
            pol_loss = update_pol(pol, optim_pol, batch)
            vf_loss = update_vf(vf, optim_vf, batch)

            pol_losses.append(pol_loss)
            vf_losses.append(vf_loss)
    else:
        for batch in traj.iterate(batch_size, epoch):
            pol_loss = update_pol(pol, optim_pol, batch)
            vf_loss = update_vf(vf, optim_vf, batch)

            pol_losses.append(pol_loss)
            vf_losses.append(vf_loss)
    logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses, VfLoss=vf_losses)

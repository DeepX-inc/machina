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

from machina import  loss_functional as lf

def update_pol(pol, optim_pol, batch, deterministic=False):
    if deterministic:
        pol_loss = lf.mse(pol, batch)
    else:
        pol_loss = lf.likelihood(pol, batch)
    pol_loss.backward()
    optim_pol.zero_grad()
    optim_pol.step()
    return pol_loss.detach().cpu().numpy()

def train(expert_traj, pol, optim_pol, batch_size, deterministic=False):
    pol_losses = []
    iterater = expert_traj.iterate_once(batch_size)
    for batch in iterater:
        pol_loss = update_pol(pol, optim_pol, batch, deterministic)
        pol_losses.append(pol_loss)
    return dict(PolLoss=pol_losses)

@torch.no_grad()
def test(expert_traj, pol, deterministic=False):
    pol.eval()
    iterater = expert_traj.full_batch(epoch=1)
    for batch in iterater:
        if deterministic:
            pol_loss = lf.mse(pol, batch)
        else:
            pol_loss = lf.likelihood(pol, batch)
    return dict(TestPolLoss=float(pol_loss.detach().cpu().numpy()))

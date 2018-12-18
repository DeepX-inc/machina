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
# This is an implementation of Stochastic Value Gradient.
# See https://arxiv.org/abs/1510.09142


import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina.misc import logger


def train(off_traj,
        pol, targ_pol, qf, targ_qf,
        optim_pol, optim_qf,
        epoch, batch_size,# optimization hypers
        tau, gamma, # advantage estimation
        sampling,
        ):

    pol_losses = []
    qf_losses = []
    logger.log("Optimizing...")
    for batch in off_traj.iterate(batch_size, epoch):
        qf_bellman_loss = lf.bellman(qf, targ_qf, targ_pol, batch, gamma, deterministic=False, sampling=sampling)
        optim_qf.zero_grad()
        qf_bellman_loss.backward()
        optim_qf.step()

        pol_loss = lf.svg(pol, qf, batch, sampling)
        optim_pol.zero_grad()
        pol_loss.backward()
        optim_pol.step()

        for q, targ_q, p, targ_p in zip(qf.parameters(), targ_qf.parameters(), pol.parameters(), targ_pol.parameters()):
            targ_p.detach().copy_((1 - tau) * targ_p.detach() + tau * p.detach())
            targ_q.detach().copy_((1 - tau) * targ_q.detach() + tau * q.detach())
        qf_losses.append(qf_bellman_loss.detach().cpu().numpy())
        pol_losses.append(pol_loss.detach().cpu().numpy())

    logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses,
        QfLoss=qf_losses,
    )


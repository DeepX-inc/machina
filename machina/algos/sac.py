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


import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina.misc import logger


def train(off_traj,
        pol, qf, targ_qf, log_alpha,
        optim_pol, optim_qf, optim_alpha,
        epoch, batch_size,# optimization hypers
        tau, gamma, sampling,
        ):

    qf_losses = []
    pol_losses = []
    alpha_losses = []
    logger.log("Optimizing...")
    for batch in off_traj.random_batch(batch_size, epoch):
        pol_loss, qf_loss, alpha_loss = lf.sac(pol, qf, targ_qf, log_alpha, batch, gamma, sampling)

        optim_pol.zero_grad()
        pol_loss.backward()
        optim_pol.step()

        optim_qf.zero_grad()
        qf_loss.backward()
        optim_qf.step()

        optim_alpha.zero_grad()
        alpha_loss.backward()
        optim_alpha.step()

        for q, targ_q in zip(qf.parameters(), targ_qf.parameters()):
            targ_q.detach().copy_((1 - tau) * targ_q.detach() + tau * q.detach())

        pol_losses.append(pol_loss.detach().cpu().numpy())
        qf_losses.append(qf_loss.detach().cpu().numpy())
        alpha_losses.append(alpha_loss.detach().cpu().numpy())

    logger.log("Optimization finished!")

    return dict(
        PolLoss=pol_losses,
        QfLoss=qf_losses,
        AlphaLoss=alpha_losses
    )


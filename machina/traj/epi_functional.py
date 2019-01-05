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


import numpy as np
import torch

from machina.utils import get_device
from machina import loss_functional as lf


def compute_vs(data, vf):
    epis = data.current_epis
    vf.reset()
    with torch.no_grad():
        for epi in epis:
            if vf.rnn:
                obs = torch.tensor(epi['obs'], dtype=torch.float, device=get_device()).unsqueeze(1)
            else:
                obs = torch.tensor(epi['obs'], dtype=torch.float, device=get_device())
            epi['vs'] = vf(obs)[0].detach().cpu().numpy()

    return data

def compute_pris(data, qf, targ_qf, targ_pol, gamma, continuous=True, deterministic=True, sampling=1, alpha=0.6, epsilon=1e-6):
    if continuous:
        epis = data.current_epis
        for epi in epis:
            data_map = dict()
            keys = ['obs', 'acs', 'rews', 'next_obs', 'dones']
            for key in keys:
                data_map[key] = torch.tensor(epi[key], device=get_device())
            with torch.no_grad():
                td_loss = lf.bellman(qf, targ_qf, targ_pol, data_map, gamma, continuous, deterministic, sampling, reduction='none')
                pris = (torch.abs(td_loss) + epsilon) ** alpha
                epi['pris'] = pris.cpu().numpy()
        return data
    else:
        raise NotImplementedError("Only Q function with continuous action space is supported now.")

def compute_rets(data, gamma):
    epis = data.current_epis
    for epi in epis:
        rews = epi['rews']
        rets = np.empty(len(rews), dtype=np.float32)
        last_rew = 0
        for t in reversed(range(len(rews))):
            rets[t] = last_rew = rews[t] + gamma * last_rew
        epi['rets'] = rets

    return data

def compute_advs(data, gamma, lam):
    epis = data.current_epis
    for epi in epis:
        rews = epi['rews']
        vs = epi['vs']
        vs = np.append(vs, 0)
        advs = np.empty(len(rews), dtype=np.float32)
        last_gaelam = 0
        for t in reversed(range(len(rews))):
            delta = rews[t] + gamma * vs[t + 1] - vs[t]
            advs[t] = last_gaelam = delta + gamma * lam * last_gaelam
        epi['advs'] = advs

    return data

def centerize_advs(data, eps=1e-6):
    epis = data.current_epis
    _advs = np.concatenate([epi['advs'] for epi in epis])
    for epi in epis:
        epi['advs'] = (epi['advs'] - np.mean(_advs)) / (np.std(_advs) + eps)

    return data

def add_next_obs(data):
    epis = data.current_epis
    for epi in epis:
        obs = epi['obs']
        _obs = [ob for ob in obs]
        next_obs = np.array(_obs[1:] + _obs[:1], dtype=np.float32)
        epi['next_obs'] = next_obs

    return data

def compute_h_masks(data):
    epis = data.current_epis
    for epi in epis:
        h_masks = np.zeros_like(epi['rews'])
        h_masks[0] = 1
        epi['h_masks'] = h_masks

    return data
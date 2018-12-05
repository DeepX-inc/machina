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

def compute_vs(data, vf):
    epis = data.current_epis
    with torch.no_grad():
        for epi in epis:
            epi['vs'] = vf(torch.tensor(epi['obs'], dtype=torch.float,
                device=get_device()))[0].cpu().numpy()

    return data

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


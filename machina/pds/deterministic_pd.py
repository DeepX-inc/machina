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
from torch.distributions import Normal, kl_divergence

from machina.pds.base import BasePd


class DeterministicPd(BasePd):
    """
    Deterministic probablistic distribution.
    """
    def sample(self, params, sample_shape=torch.Size()):
        mean = params['mean']
        ac = Normal(loc=mean, scale=torch.zeros_like(
            mean)).rsample(sample_shape)
        return ac

    def llh(self, x, params):
        mean = params['mean']
        return torch.sum(Normal(loc=mean, scale=torch.zeros_like(mean)).log_prob(x), dim=-1)

    def kl_pq(self, p_params, q_params):
        p_mean = p_params['mean']
        q_mean = q_params['mean']
        return torch.sum(kl_divergence(Normal(loc=p_mean, scale=torch.zeros_like(p_mean)), Normal(loc=q_mean, scale=torch.zeros_like(q_mean))), dim=-1)

    def ent(self, params):
        mean = params['mean']
        return torch.sum(Normal(loc=mean, scale=torch.zeros_like(mean)).entropy(), dim=-1)

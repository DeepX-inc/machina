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
from torch.distributions import Categorical, kl_divergence

from machina.pds.base import BasePd

class MultiCategoricalPd(BasePd):
    def __init__(self, ob_space, ac_space):
        BasePd.__init__(self, ob_space, ac_space)

    def sample(self, params):
        pis = params['pis']
        pis_sampled = []
        for pi in torch.chunk(pis, pis.size(-2), -2):
            pi_sampled = Categorical(probs=pi).sample()
            pis_sampled.append(pi_sampled)
        return torch.cat(pis_sampled, dim=-1)

    def llh(self, xs, params):
        pis = params['pis']
        llhs = []
        for x, pi in zip(torch.chunk(xs, xs.size(-1), -1), torch.chunk(pis, pis.size(-2), -2)):
            x = x.squeeze(-1)
            pi = pi.squeeze(-2)
            llhs.append(Categorical(pi).log_prob(x))
        return sum(llhs)

    def kl_pq(self, p_params, q_params):
        p_pis = p_params['pis']
        q_pis = q_params['pis']
        kls = []
        for p_pi, q_pi in zip(torch.chunk(p_pis, pi_pis.size(-2), -2), torch.chunk(q_pis, q_pis.size(-2), -2)):
            kls.append(kl_divergence(Categorical(p_pi), Categorical(q_pi)))
        return sum(kls)

    def ent(self, params):
        pis = params['pis']
        ents = []
        for pi in torch.chunk(pis, pis.size(-2), -2):
            ents.append(Categorical(pi).entropy())
        return sum(ents)

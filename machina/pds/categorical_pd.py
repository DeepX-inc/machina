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

class CategoricalPd(BasePd):
    def __init__(self, ob_space, ac_space):
        BasePd.__init__(self, ob_space, ac_space)

    def sample(self, params):
        pi = params['pi']
        pi_sampled = Categorical(probs=pi).sample()
        return pi_sampled

    def llh(self, x, params):
        pi = params['pi']
        return Categorical(pi).log_prob(x)

    def kl_pq(self, p_params, q_params):
        p_pi = p_params['pi']
        q_pi = q_params['pi']
        return kl_divergence(Categorical(p_pi), Categorical(q_pi))

    def ent(self, params):
        pi = params['pi']
        return Categorical(pi).entropy()



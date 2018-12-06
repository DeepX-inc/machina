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

import copy
import numpy as np
import torch
from machina.utils import cpu_mode
from machina.samplers.base import BaseSampler, one_epi


class BatchSampler(BaseSampler):
    def __init__(self, env):
        BaseSampler.__init__(self, env)

    def sample(self, pol, max_samples, max_episodes, deterministic=False, prepro=None):
        sampling_pol = copy.deepcopy(pol)
        sampling_pol = sampling_pol.cpu()
        sampling_pol.eval()
        n_samples = 0
        n_episodes = 0
        epis = []
        with cpu_mode():
            while max_samples > n_samples and max_episodes > n_episodes:
                l, epi = one_epi(self.env, sampling_pol, deterministic, prepro)
                n_samples += l
                n_episodes += 1
                epis.append(epi)
        return epis

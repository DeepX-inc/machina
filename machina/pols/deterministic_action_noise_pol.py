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
from machina.pols import BasePol
from machina.utils import get_device

class DeterministicActionNoisePol(BasePol):
    def __init__(self, ob_space, ac_space, net, noise=None, normalize_ac=True):
        BasePol.__init__(self, ob_space, ac_space, normalize_ac)
        self.net = net
        self.noise = noise
        self.to(get_device())

    def reset(self):
        super(DeterministicActionNoisePol, self).reset()
        if self.noise is not None:
            self.noise.reset()
        else:
            pass

    def forward(self, obs):
        mean = self.net(obs)
        ac = mean[0]
        if self.noise is not None:
            action_noise = self.noise()
            ac = ac + action_noise
        else:
            pass
        ac_real = self.convert_ac_for_real(ac.detach().cpu().numpy())
        return ac_real, ac, dict(mean=mean[0])

    def deterministic_ac_real(self, obs):
        """
        action for deployment
        """
        mean = self.net(obs)[0]
        mean_real = self.convert_ac_for_real(mean.detach().cpu().numpy())
        return mean_real

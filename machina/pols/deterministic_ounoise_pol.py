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

class ActionNoise(object):
    def reset(self):
        pass


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma * np.ones_like(self.mu)
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return torch.tensor(x, detype=torch.float32, device=get_device())

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mu, dtype = np.float32)


class DeterministicPol(BasePol):
    def __init__(self, ob_space, ac_space, net, noise=None, apply_noise=False, normalize_ac=True):
        BasePol.__init__(self, ob_space, ac_space, normalize_ac)
        self.net = net
        self.to(get_device())
        self.noise = noise
        self.apply_noise = apply_noise

    def reset(self):
        if self.noise is not None:
            self.noise.reset()
        else:
            pass

    def forward(self, obs):
        mean = self.net(obs)
        ac = mean
        apply_noise = self.apply_noise
        if self.noise is not None and apply_noise:
            action_noise = self.noise()
            ac = ac + action_noise
        else:
            pass
        ac_real = self.convert_ac_for_real(ac.detach().cpu().numpy())
        return ac_real, ac, dict(mean=mean)
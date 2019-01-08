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


import gym
import numpy as np


class C2DEnv(object):
    def __init__(self, env, n_bins=30):
        assert isinstance(env.ac_space, gym.spaces.Box)
        assert len(env.ac_space.shape) == 1
        self.env = env
        self.n_bins = n_bins
        self.ac_space = gym.spaces.MultiDiscrete(
            env.ac_space.shape[0] * [n_bins])

    @property
    def observation_space(self):
        return self.env.ob_space

    @property
    def action_space(self):
        return self.ac_space

    @property
    def horizon(self):
        return self.env._horizon

    def reset(self):
        return self.env.reset()

    def step(self, action):
        continuous_action = []
        for a, low, high in zip(action, self.env.ac_space.low, self.env.ac_space.high):
            continuous_action.append(np.linspace(low, high, self.n_bins)[a])
        action = np.array(continuous_action)
        next_obs, reward, done, info = self.env.step(action)
        return next_obs, reward, done, info

    def render(self):
        self.env.render()

    def terminate(self):
        self.env.terminate()

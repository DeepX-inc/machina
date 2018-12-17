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

import unittest

import gym
import numpy as np
import torch

from machina.envs import GymEnv, C2DEnv


def test_continuous2discrete():
    continuous_env = GymEnv('LunarLanderContinuous-v2', record_video=False)
    discrete_env = C2DEnv(continuous_env)

    discrete_env.reset()
    out = discrete_env.step([3, 10])

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


import torch.nn as nn


class BaseSVfunc(nn.Module):
    """
    Base function of State Value Function.
    It takes observations and then output value.
    For example V Func.

    Parameters
    ----------
    ob_space : gym.Space
    net : torch.nn.Module
    rnn : bool
    data_parallel : bool
        If True, network computation is executed in parallel.
    parallel_dim : int
        Splitted dimension in data parallel.
    """
    def __init__(self, ob_space, net, rnn=False, data_parallel=False, parallel_dim=0):
        nn.Module.__init__(self)
        self.ob_space = ob_space
        self.net = net

        self.rnn = rnn
        self.hs = None

        self.data_parallel = data_parallel
        if data_parallel:
            self.dp_net = nn.DataParallel(self.net, dim=parallel_dim)
        self.dp_run = False

    def reset(self):
        """
        reset for rnn's hidden state.
        """
        if self.rnn:
            self.hs = None

    def _check_obs_shape(self, obs):
        """
        Reshape input appropriately.
        """
        if self.rnn:
            additional_shape = 2
        else:
            additional_shape = 1
        if len(obs.shape) < additional_shape + len(self.ob_space.shape):
            for _ in range(additional_shape + len(self.ob_space.shape) - len(obs.shape)):
                obs.unsqueeze(0)
        return obs

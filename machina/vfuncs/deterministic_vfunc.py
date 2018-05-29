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

import torch

from machina.vfuncs.base import BaseVfunc
from machina.utils import get_gpu, Variable

class DeterministicVfunc(BaseVfunc):
    def __init__(self, ob_space, net):
        BaseVfunc.__init__(self, ob_space)
        self.net = net
        gpu_id = get_gpu()
        if gpu_id != -1:
            self.cuda(gpu_id)

    def forward(self, obs):
        return self.net(obs).view(-1)


class NormalizedDeterministicVfunc(DeterministicVfunc):
    def __init__(self, ob_space, net):
        DeterministicVfunc.__init__(self, ob_space, net)
        self.x_mean = Variable(torch.zeros(1))
        self.x_std = Variable(torch.ones(1))

    def forward(self, obs):
        return self.net(obs).view(-1) * self.x_std + self.x_mean

    def set_mean(self, mean):
        self.x_mean.data.copy_(mean)

    def set_std(self, std):
        self.x_std.data.copy_(std)



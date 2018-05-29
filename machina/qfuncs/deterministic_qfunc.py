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

from machina.qfuncs.base import BaseQfunc
from machina.utils import get_gpu


class DeterministicQfunc(BaseQfunc):
    def __init__(self, ob_space, ac_space, net):
        BaseQfunc.__init__(self, ob_space, ac_space)
        self.net = net
        gpu_id = get_gpu()
        if gpu_id != -1:
            self.cuda(gpu_id)

    def forward(self, obs, acs):
        return self.net(obs, acs).view(-1)


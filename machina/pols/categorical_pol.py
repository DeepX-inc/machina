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
import torch.nn as nn
from machina.pols import BasePol
from machina.pds.categorical_pd import CategoricalPd
from machina.utils import get_device

class CategoricalPol(BasePol):
    def __init__(self, ob_space, ac_space, net, rnn=False, normalize_ac=True, data_parallel=False, parallel_dim=0):
        BasePol.__init__(self, ob_space, ac_space, net, rnn,  normalize_ac, data_parallel, parallel_dim)
        self.pd = CategoricalPd(ob_space, ac_space)
        self.to(get_device())

    def forward(self, obs, hs=None, masks=None):
        if self.rnn:
            time_seq, batch_size, *_ = obs.shape

            if hs is None:
                if self.hs is None:
                    self.hs = self.net.init_hs(batch_size)
                hs = self.hs

            if masks is None:
                masks = hs[0].new(time_seq, batch_size, 1).zero_()
            masks = masks.reshape(time_seq, batch_size, 1)

            if self.dp_run:
                pi, hs = self.dp_net(obs, hs, masks)
            else:
                pi, hs = self.net(obs, hs, masks)
            self.hs = hs
        else:
            if self.dp_run:
                pi = self.dp_net(obs)
            else:
                pi = self.net(obs)
        ac = self.pd.sample(dict(pi=pi))
        ac_real = self.convert_ac_for_real(ac.detach().cpu().numpy())
        return ac_real, ac, dict(pi=pi, hs=hs)

    def deterministic_ac_real(self, obs, hs=None, mask=None):
        """
        action for deployment
        """
        if self.rnn:
            time_seq, batch_size, *_ = obs.shape
            if hs is None:
                if self.hs is None:
                    self.hs = self.init_hs(batch_size)
                hs = self.hs
            pi, hs = self.net(obs, hs, mask)
            self.hs = hs
        else:
            pi = self.net(obs)
        _, ac = torch.max(pi, dim=-1)
        ac_real = self.convert_ac_for_real(ac.detach().cpu().numpy())
        return ac_real, ac, dict(pi=pi, hs=hs)

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

from machina.pds.base import BasePd

class GaussianPd(BasePd):
    def __init__(self, ob_space, ac_space):
        BasePd.__init__(self, ob_space, ac_space)

    def sample(self, params):
        mean, log_std = params['mean'], params['log_std']
        ac = mean + torch.randn_like(mean) * torch.exp(log_std)
        return ac

    def llh(self, x, params):
        mean, log_std = params['mean'], params['log_std']
        std = torch.exp(log_std)
        return - 0.5 * torch.sum(((x - mean) / std)**2, dim=-1) \
                - 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                - torch.sum(log_std, dim=-1)

    def kl_pq(self, p_params, q_params):
        p_mean, p_log_std = p_params['mean'], p_params['log_std']
        q_mean, q_log_std = q_params['mean'], q_params['log_std']
        p_std = torch.exp(p_log_std)
        q_std = torch.exp(q_log_std)
        return torch.sum(((p_mean - q_mean)**2 + p_std**2 - q_std**2) / (2 * q_std**2 + 1e-8) + q_log_std - p_log_std, dim=-1)

    def ent(self, params):
        log_std = params['log_std']
        return torch.sum(log_std + 0.5 * torch.tensor(np.log(2.0 * np.pi * np.e), dtype=torch.float, device=log_std.device), dim=-1)


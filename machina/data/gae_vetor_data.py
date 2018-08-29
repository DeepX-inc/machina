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
import scipy

from machina.data.base import BaseData
from machina.utils import get_device

class GAEVectorData(BaseData):
    def __init__(self, paths, rnn=True):
        self.paths = paths
        self.data_map = {}
        self.n = sum([len(path['rews']) for path in paths])
        self._next_id = 0
        self.num_epi = len(paths)
        self.num_step = len(paths[0]['rews'])
        self.num_env = len(paths)
        self.rnn = rnn

    def preprocess(self, vf, gamma, lam, centerize=True):
        with torch.no_grad():
            obs_shape = self.paths[0]['obs'][0].shape
            obs = torch.tensor([path['obs'] for path in self.paths], dtype=torch.float, device=get_device()).transpose(1, 0)
            init_hs = torch.cat(
                [torch.tensor(np.concatenate([path['init_hs'][0] for path in self.paths], axis=0), dtype=torch.float, device=get_device()).unsqueeze(0),
                torch.tensor(np.concatenate([path['init_hs'][1] for path in self.paths], axis=0), dtype=torch.float, device=get_device()).unsqueeze(0)]
            )
            dones = torch.tensor(np.array([path['dones'] for path in self.paths]), dtype=torch.float, device=get_device()).t()
            vs, hs = vf(obs, init_hs, dones)

            last_obs = torch.tensor([path['last_ob'] for path in self.paths], dtype=torch.float, device=get_device()).unsqueeze(0)
            last_d = torch.tensor(np.array([path['last_d'] for path in self.paths]), dtype=torch.float, device=get_device()).squeeze()
            last_vs, _ = vf(last_obs, hs, last_d.unsqueeze(0))

            advs = torch.zeros_like(vs)
            rets = torch.zeros_like(vs)

            rews = torch.tensor([path['rews'] for path in self.paths], dtype=torch.float, device=get_device()).t()

            last_gaelam = 0
            for t in reversed(range(self.num_step)):
                if t == self.num_step - 1:
                    next_nonterminal = 1.0 - last_d
                    next_vs = last_vs
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = vs[t+1]
                delta = rews[t] + gamma * next_vs * next_nonterminal - vs[t]
                advs[t] = last_gaelam = delta + gamma * lam * next_nonterminal * last_gaelam
            rets = advs + vs

            self.data_map['obs'] = obs
            self.data_map['acs'] = torch.tensor([path['acs'] for path in self.paths], dtype=torch.float, device=get_device()).transpose(1, 0)
            self.data_map['rews'] = rews.detach()
            self.data_map['dones'] = dones.detach()
            self.data_map['advs'] = advs.detach()
            self.data_map['rets'] = rets.detach()
            self.data_map['init_hs'] = init_hs.detach()
            if centerize:
                self.data_map['advs'] = (self.data_map['advs'] - torch.mean(
                    self.data_map['advs'])) / (torch.std(self.data_map['advs']) + 1e-6)
            keys = self.paths[0].keys()
            for key in keys:
                if isinstance(self.paths[0][key], dict):
                    new_keys = self.paths[0][key].keys()
                    for new_key in new_keys:
                        self.data_map[new_key] = torch.tensor([path[key][new_key] for path in self.paths], dtype=torch.float, device=get_device()).transpose(1, 0)


    def shuffle(self):
        perm = np.arange(self.num_env)
        np.random.shuffle(perm)
        perm = torch.tensor(perm, dtype=torch.long, device=get_device())

        for key in self.data_map:
            self.data_map[key] = self.data_map[key][:, perm]

        self._next_id = 0

    def next_batch(self, batch_size):
        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.num_env - self._next_id)
        self._next_id += cur_batch_size

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][:, cur_id:cur_id+cur_batch_size]
        return data_map


    def iterate(self, batch_size, epoch=1):
        for _ in range(epoch):
            self.shuffle()
            while self._next_id <= self.num_env - batch_size:
                yield self.next_batch(batch_size)
            self._next_id = 0

    def full_batch(self, epoch=1):
        for _ in range(epoch):
            yield self.data_map

    def __del__(self):
        del self.paths
        del self.data_map


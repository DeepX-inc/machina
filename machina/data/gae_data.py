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


def discount_cumsum(x, discount):
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter([1], [1, float(-discount)], x.numpy()[::-1], axis=0)[::-1]


class GAEData(BaseData):
    def __init__(self, paths, shuffle=True):
        self.paths = paths
        self.data_map = {}
        self.enable_shuffle = shuffle
        self.n = sum([len(path['rews']) for path in paths])
        self._next_id = 0
        self.num_epi = len(paths)

    def path2data_map(self, centerize):
        keys = self.paths[0].keys()
        for key in keys:
            if isinstance(self.paths[0][key], list) or isinstance(self.paths[0][key], np.ndarray):
                self.data_map[key] = torch.tensor(np.concatenate(
                    [path[key] for path in self.paths], axis=0), dtype=torch.float, device=get_device())
            elif isinstance(self.paths[0][key], dict):
                new_keys = self.paths[0][key].keys()
                for new_key in new_keys:
                    self.data_map[new_key] = torch.tensor(np.concatenate(
                        [path[key][new_key] for path in self.paths], axis=0), dtype=torch.float, device=get_device())
        if centerize:
            self.data_map['advs'] = (self.data_map['advs'] - torch.mean(
                self.data_map['advs'])) / (torch.std(self.data_map['advs']) + 1e-6)

    def preprocess(self, vf, gamma, lam, centerize=True):
        with torch.no_grad():
            all_path_vs = [vf(torch.tensor(path['obs'], dtype=torch.float,
                                           device=get_device())).cpu().numpy() for path in self.paths]
        for idx, path in enumerate(self.paths):
            path_vs = np.append(all_path_vs[idx], 0)
            rews = path['rews']
            advs = np.empty(len(rews), dtype='float32')
            rets = np.empty(len(rews), dtype='float32')
            last_gaelam = 0
            last_rew = 0
            for t in reversed(range(len(rews))):
                delta = rews[t] + gamma * path_vs[t+1] - path_vs[t]
                advs[t] = last_gaelam = delta + gamma * lam * last_gaelam
                rets[t] = last_rew = rews[t] + gamma * last_rew
            path['advs'] = advs
            path['rets'] = rets
            path['vs'] = path_vs[:-1]
        self.path2data_map(centerize=centerize)

    def shuffle(self):
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        perm = torch.tensor(perm, dtype=torch.long, device=get_device())

        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]

        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id >= self.n and self.enable_shuffle:
            self.shuffle()

        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.n - self._next_id)
        self._next_id += cur_batch_size

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][cur_id:cur_id+cur_batch_size]
        return data_map

    def iterate_once(self, batch_size):
        if self.enable_shuffle:
            self.shuffle()

        while self._next_id <= self.n - batch_size:
            yield self.next_batch(batch_size)
        self._next_id = 0

    def iterate(self, batch_size, epoch=1):
        if self.enable_shuffle:
            self.shuffle()
        for _ in range(epoch):
            while self._next_id <= self.n - batch_size:
                yield self.next_batch(batch_size)
            self._next_id = 0

    def full_batch(self, epoch=1):
        if self.enable_shuffle:
            self.shuffle()
        for _ in range(epoch):
            yield self.data_map

    def __del__(self):
        del self.paths
        del self.data_map

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

from machina.utils import get_device

class Data(object):
    def __init__(self):
        self.data_map = dict()
        self._next_id = 0

        self.current_epis = None
        self._epis_index = np.array([0])

    @property
    def num_step(self):
        if self.data_map:
            return len(self.data_map[list(self.data_map.keys())[0]])
        return 0

    @property
    def num_epi(self):
        return len(self._epis_index) - 1

    def add_epis(self, epis):
        self.current_epis = epis

    def _concat_data_map(self, data_map):
        if self.data_map:
            for key in data_map:
                self.data_map[key] = torch.cat([self.data_map[key], data_map[key]], dim=0)
        else:
            self.data_map = data_map

    def register_epis(self):
        epis = self.current_epis
        keys = epis[0].keys()
        data_map = dict()
        for key in keys:
            if isinstance(epis[0][key], list) or isinstance(epis[0][key], np.ndarray):
                data_map[key] = torch.tensor(np.concatenate(
                    [epi[key] for epi in epis], axis=0), dtype=torch.float, device=get_device())
            elif isinstance(epis[0][key], dict):
                new_keys = epis[0][key].keys()
                for new_key in new_keys:
                    data_map[new_key] = torch.tensor(np.concatenate(
                        [epi[key][new_key] for epi in epis], axis=0), dtype=torch.float, device=get_device())

        self._concat_data_map(data_map)

        epis_index = []
        index = 0
        for epi in epis:
            l_epi = len(epi[list(epi.key())[0]])
            index += l_epi
            epis_index.append(index)
        epis_index = np.array(epis_index) + self._epis_index[-1]
        self._epis_index = np.concatenate([self._epis_index, epis_index])

        self.current_epis = None

    def add_data(self, data):
        self._concat_data_map(data.data_map)

        epis_index = data._epis_index
        epis_index = epis_index + self._epis_index[-1]
        self._epis_index = np.concatenate([self._epis_index, epis_index[1:]])

    def _shuffled_indices(self, indices):
        return indices[torch.randperm(len(indices), device=get_device())]

    def _get_indices(self, indices=None, shuffle=True):
        if indices is None:
            indices = torch.arange(self.num_step, device=get_device())
        if shuffle:
            indices = self._shuffled_indices(indices)
        return indices

    def _next_batch(self, batch_size, indices):
        cur_id = self._next_id
        cur_batch_size = min(batch_size, len(indices) - self._next_id)
        self._next_id += cur_batch_size

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][cur_id:cur_id+cur_batch_size]
        return data_map

    def iterate_once(self, batch_size, indices=None, shuffle=True):
        indices = self._get_indices(indices, shuffle)

        while self._next_id <= len(indices) - batch_size:
            yield self._next_batch(batch_size, indices)
        self._next_id = 0

    def iterate(self, batch_size, epoch=1, indices=None, shuffle=True):
        indices = self._get_indices(indices, shuffle)

        for _ in range(epoch):
            while self._next_id <= len(indices) - batch_size:
                yield self._next_batch(batch_size, indices)
            self._next_id = 0

    def random_batch_once(self, batch_size, indices=None):
        indices = self._get_indices(indices, shuffle=True)

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][indices[:batch_size]]
        return data_map

    def random_batch(self, batch_size, epoch=1, indices=None):
        for _ in range(epoch):
            batch = self.random_batch_once(batch_size, indices)
            yield batch

    def full_batch(self, epoch=1):
        for _ in range(epoch):
            yield self.data_map

    def iterate_epi(self):
        for i in range(len(self._epis_index) - 1):
            yield self.data_map[self._epis_index[i]:self._epis_index[i+1]]

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

import copy

import gym
import numpy as np
import torch
import torch.multiprocessing as mp

from machina.utils import cpu_mode
from machina.samplers.base import BaseSampler, one_epi


def sample_process(pol, env, max_samples, max_episodes, n_samples_global, n_episodes_global, epis, exec_flags, deterministic_flag, process_id, prepro=None, seed=256):

    np.random.seed(seed + process_id)
    torch.manual_seed(seed + process_id)
    torch.set_num_threads(1)

    while True:
        if exec_flags[process_id] > 0:
            while max_samples > n_samples_global and max_episodes > n_episodes_global:
                l, epi = one_epi(env, pol, deterministic_flag, prepro)
                n_samples_global += l
                n_episodes_global += 1
                epis.append(epi)
            exec_flags[process_id].zero_()

class ParallelSampler(BaseSampler):
    def __init__(self, env, pol, max_samples, max_episodes, num_parallel=8, prepro=None, seed=256):
        BaseSampler.__init__(self, env)
        self.pol = copy.deepcopy(pol)
        self.pol.to('cpu')
        self.pol.share_memory()
        self.pol.eval()
        self.max_samples = max_samples
        self.max_episodes = max_episodes
        self.num_parallel = num_parallel

        self.n_samples_global = torch.tensor(0, dtype=torch.long).share_memory_()
        self.n_episodes_global = torch.tensor(0, dtype=torch.long).share_memory_()
        self.exec_flags = [torch.tensor(0, dtype=torch.long).share_memory_() for _ in range(self.num_parallel)]
        self.deterministic_flag = torch.tensor(0, dtype=torch.uint8).share_memory_()

        self.epis = mp.Manager().list()
        self.processes = []
        for ind in range(self.num_parallel):
            p = mp.Process(target=sample_process, args=(self.pol, env, max_samples, max_episodes, self.n_samples_global, self.n_episodes_global, self.epis, self.exec_flags, self.deterministic_flag, ind, prepro, seed))
            p.start()
            self.processes.append(p)

    def __del__(self):
        for p in self.processes:
            p.terminate()

    def sample(self, pol, *args, **kwargs):
        deterministic = kwargs.pop('deterministic', False)
        if deterministic:
            self.deterministic_flag.zero_()
            self.deterministic_flag += 1
        else:
            self.deterministic_flag.zero_()

        for sp, p in zip(self.pol.parameters(), pol.parameters()):
            sp.data.copy_(p.data.to('cpu'))

        self.n_samples_global.zero_()
        self.n_episodes_global.zero_()

        del self.epis[:]

        for exec_flag in self.exec_flags:
            exec_flag += 1

        while True:
            if all([exec_flag == 0 for exec_flag in self.exec_flags]):
                return list(self.epis)


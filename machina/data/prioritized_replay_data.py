import numpy as np
import torch

from machina.data.base import BaseData
from machina.utils import np2torch, torch2torch

class PrioritizedReplayData(BaseData):
    def __init__(
            self, max_data_size, ob_dim, ac_dim, rew_scale=1):
        self.max_data_size = max_data_size
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.rew_scale = rew_scale


        self.obs = torch2torch(torch.zeros((max_data_size, ob_dim))).float()
        self.acs = torch2torch(torch.zeros((max_data_size, ac_dim))).float()
        self.rews = torch2torch(torch.zeros(max_data_size)).float()
        self.delta = torch2torch(torch.zeros((max_data_size)).float())
        self.rank = torch2torch(torch.zeros((max_data_size)).float())
        self.terminals = torch2torch(torch.zeros(max_data_size)).float()
        self.bottom = 0
        self.top = 0
        self.size = 0

    def add_sample(self, ob, ac, rew, next_ob, terminal):
        self.obs[self.top] = np2torch(ob).float()
        self.acs[self.top] = np2torch(ac).float()
        self.rews[self.top] = rew * self.rew_scale
        self.delta[self.top] = torch.mean(np2torch(((ob-next_ob)**2)))
        self.terminals[self.top] = terminal
        self.top = (self.top + 1) % self.max_data_size
        if self.size >= self.max_data_size:
            self.bottom = (self.bottom + 1) % self.max_data_size
        else:
            self.size += 1

    def add_path(self, path):
        next_obs = np.append(path['obs'][1:], np.array([path['obs'][-1]]))
        for i, (ob, ac, rew, next_ob) in enumerate(zip(path['obs'], path['acs'], path['rews'], next_obs)):
            if i == len(path['rews']) - 1:
                self.add_sample(ob, ac, rew, next_ob, 1)
            else:
                self.add_sample(ob, ac, rew, next_ob, 0)

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    def iterate_once(self, batch_size):
        assert self.size > batch_size
        indices = torch2torch(torch.zeros(batch_size)).long()
        transition_indices = torch2torch(torch.zeros(batch_size)).long()
        sum_delta = torch2torch(torch.sum(self.delta))
        rand_list = np2torch(np.random.uniform(0, sum_delta, batch_size))
        rand_list = torch.sort(rand_list)

        idx = -1
        tmp_sum_delta = 0
        for (i, randnum) in enumerate(rand_list):
            while tmp_sum_delta <= randnum:
                tmp_sum_delta += self.delta[idx] + 0.0001
                idx += 1

            indices[i] = idx
            transition_indices[i] = (idx + 1) % self.max_data_size


        return dict(
            obs=self.obs[indices],
            acs=self.acs[indices],
            rews=self.rews[indices],
            terminals=self.terminals[indices],
            next_obs=self.obs[transition_indices]
        )

    def iterate(self, batch_size, epoch=1):
        for _ in range(epoch):
            batch = self.iterate_once(batch_size)
            yield batch


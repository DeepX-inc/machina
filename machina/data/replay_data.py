import numpy as np
import torch

from machina.data.base import BaseData
from machina.utils import np2torch, torch2torch

class ReplayData(BaseData):
    def __init__(
            self, max_data_size, ob_dim, ac_dim, rew_scale=1):
        self.max_data_size = max_data_size
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.rew_scale = rew_scale

        self.obs = torch2torch(torch.zeros((max_data_size, ob_dim))).float()
        self.acs = torch2torch(torch.zeros((max_data_size, ac_dim))).float()
        self.rews = torch2torch(torch.zeros(max_data_size)).float()
        self.terminals = torch2torch(torch.zeros(max_data_size)).float()
        self.bottom = 0
        self.top = 0
        self.size = 0

    def add_sample(self, ob, ac, rew, terminal):
        self.obs[self.top] = np2torch(ob).float()
        self.acs[self.top] = np2torch(ac).float()
        self.rews[self.top] = rew * self.rew_scale
        self.terminals[self.top] = terminal
        self.top = (self.top + 1) % self.max_data_size
        if self.size >= self.max_data_size:
            self.bottom = (self.bottom + 1) % self.max_data_size
        else:
            self.size += 1

    def add_path(self, path):
        for i, (ob, ac, rew) in enumerate(zip(path['obs'], path['acs'], path['rews'])):
            if i == len(path['rews']) - 1:
                self.add_sample(ob, ac, rew, 1)
            else:
                self.add_sample(ob, ac, rew, 0)

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    def iterate_once(self, batch_size):
        assert self.size > batch_size
        indices = torch2torch(torch.zeros(batch_size)).long()
        transition_indices = torch2torch(torch.zeros(batch_size)).long()
        count = 0
        while count < batch_size:
            index = np.random.randint(self.bottom, self.bottom + self.size) % self.max_data_size
            # make sure that the transition is valid: if we are at the end of the data, we need to discard
            # this sample
            if index == self.size - 1 and self.size <= self.max_data_size:
                continue
            # if self._terminals[index]:
            #     continue
            transition_index = (index + 1) % self.max_data_size
            indices[count] = index
            transition_indices[count] = transition_index
            count += 1
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


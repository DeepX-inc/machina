import numpy as np
import torch
from ..utils import Variable
import scipy

from .base import BaseData

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
                self.data_map[key] = np.concatenate([path[key] for path in self.paths], axis=0)
            elif isinstance(self.paths[0][key], dict):
                new_keys = self.paths[0][key].keys()
                for new_key in new_keys:
                    self.data_map[new_key] = np.concatenate([path[key][new_key] for path in self.paths], axis=0)
        if centerize:
            self.data_map['advs'] = (self.data_map['advs'] - np.mean(self.data_map['advs'])) / (np.std(self.data_map['advs']) + 1e-6)

    def preprocess(self, vf, gamma, lam, centerize=True):
        all_path_vs = [vf(Variable(torch.from_numpy(path['obs']).float(), volatile=True)).data.cpu().numpy() for path in self.paths]
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
        if self.enable_shuffle: self.shuffle()

        while self._next_id <= self.n - batch_size:
            yield self.next_batch(batch_size)
        self._next_id = 0

    def iterate(self, batch_size, epoch=1):
        if self.enable_shuffle: self.shuffle()
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




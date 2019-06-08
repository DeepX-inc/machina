"""
trajectory class
"""

import functools

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data

from machina import loss_functional as lf
from machina.utils import get_device

LARGE_NUMBER = 1000000000000


class Traj(object):
    """
    Trajectory class.
    A Trajectory is a sequence of episodes.
    An episode is a sequence of steps.

    This class provides batch methods.

    Parameters
    ----------
    max_steps: None or int
        Specifying maximum steps to be saved in Traj.
    traj_device: None or str or torch.device
        Device name Traj is allocated.
    """

    def __init__(self, max_steps=None, traj_device=None, ddp=False):
        self.data_map = dict()
        self._next_id = 0

        self.current_epis = None
        self._epis_index = np.array([0])

        self.max_steps = max_steps if max_steps is not None else LARGE_NUMBER

        if traj_device is None:
            self.traj_device = lambda: get_device()
        else:
            self.traj_device = lambda: traj_device

        self.ddp = ddp
        if ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    @property
    def num_step(self):
        return int(self._epis_index[-1])

    @property
    def num_epi(self):
        return len(self._epis_index) - 1

    def to(self, device):
        """Perform data_map tensor type conversion
        """
        for key in self.data_map:
            self.data_map[key] = self.data_map[key].to(device)

    def copy(self, traj):
        self.data_map = traj.data_map
        self._next_id = traj._next_id
        self.current_epis = traj.current_epis
        self._epis_index = traj._epis_index
        self.max_steps = traj.max_steps
        self.traj_device = traj.traj_device
        if hasattr(self, 'pri_beta') and hasattr(traj, 'pri_beta'):
            self.pri_beta = traj.pri_beta

    def get_max_pri(self):
        if 'pris' in self.data_map:
            return torch.max(self.data_map['pris']).cpu()
        else:
            return torch.tensor(100)

    def add_epis(self, epis):
        self.current_epis = epis

    def _concat_data_map(self, data_map, remain_index=None):
        if self.data_map:
            for key in data_map:
                if remain_index is not None:
                    self.data_map[key] = torch.cat(
                        [self.data_map[key][self._epis_index[remain_index]:], data_map[key].to(self.traj_device())], dim=0)
                else:
                    self.data_map[key] = torch.cat(
                        [self.data_map[key], data_map[key].to(self.traj_device())], dim=0)
        else:
            self.data_map = dict()
            for key in data_map:
                self.data_map[key] = data_map[key].to(self.traj_device())

    def register_epis(self):
        epis = self.current_epis
        keys = epis[0].keys()
        data_map = dict()
        for key in keys:
            if isinstance(epis[0][key], list) or isinstance(epis[0][key], np.ndarray):
                data_map[key] = torch.tensor(np.concatenate(
                    [epi[key] for epi in epis], axis=0), dtype=torch.float, device=self.traj_device())
            elif isinstance(epis[0][key], dict):
                new_keys = epis[0][key].keys()
                for new_key in new_keys:
                    data_map[new_key] = torch.tensor(np.concatenate(
                        [epi[key][new_key] for epi in epis], axis=0), dtype=torch.float, device=self.traj_device())

        self._concat_data_map(data_map)

        epis_index = []
        index = 0
        for epi in epis:
            l_epi = len(epi['rews'])
            index += l_epi
            epis_index.append(index)
        epis_index = np.array(epis_index) + self._epis_index[-1]
        self._epis_index = np.concatenate([self._epis_index, epis_index])

        self.current_epis = None

    def add_traj(self, traj):
        epis_index = traj._epis_index
        pre_num_step = self.num_step
        add_num_step = traj.num_step
        if pre_num_step + add_num_step <= self.max_steps:
            self._concat_data_map(traj.data_map)
            epis_index = epis_index + self._epis_index[-1]
            self._epis_index = np.concatenate(
                [self._epis_index, epis_index[1:]])
        elif add_num_step <= self.max_steps:
            remain_index = 0
            while self.max_steps < pre_num_step + add_num_step - self._epis_index[remain_index]:
                remain_index += 1
            self._concat_data_map(traj.data_map, remain_index)
            self._epis_index = self._epis_index[remain_index:] - \
                self._epis_index[remain_index]
            epis_index = epis_index + self._epis_index[-1]
            self._epis_index = np.concatenate(
                [self._epis_index, epis_index[1:]])
        else:  # self.max_steps < add_step
            remain_index = -1
            while self.max_steps < epis_index[remain_index]:
                if remain_index == - len(epis_index):
                    raise ValueError(
                        'max_steps should be larger than the number of steps in one episode.')
                remain_index -= 1
            for key in traj.data_map:
                self.data_map[key] = traj.data_map[key][:epis_index[remain_index]]
            self._epis_index = traj._epis_index[:remain_index+1]

    def _shuffled_indices(self, indices):
        return indices[torch.randperm(len(indices))]

    def _get_indices(self, indices=None, shuffle=True):
        if indices is None:
            indices = torch.arange(
                self.num_step, dtype=torch.long)
        if shuffle:
            indices = self._shuffled_indices(indices)
        return indices

    def _next_batch(self, batch_size, indices):
        cur_id = self._next_id
        cur_batch_size = min(batch_size, len(indices) - self._next_id)
        self._next_id += cur_batch_size

        if self.ddp:
            indices = indices[cur_id + self.rank:cur_id +
                              cur_batch_size:self.world_size]
        else:
            indices = indices[cur_id:cur_id + batch_size]

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][indices].to(get_device())
        return data_map

    def iterate_once(self, batch_size, indices=None, shuffle=True):
        """
        Iterate a full of trajectory once.

        Parameters
        ----------
        batch_size : int
        indices : ndarray or torch.Tensor or None
            Selected indices for iteration.
            If None, whole trajectory is selected.
        shuffle : bool

        Returns
        -------
        data_map : dict of torch.Tensor
        """
        indices = self._get_indices(indices, shuffle)

        while self._next_id <= len(indices) - batch_size:
            yield self._next_batch(batch_size, indices)
        self._next_id = 0

    def iterate(self, batch_size, epoch=1, indices=None, shuffle=True):
        """
        Iterate a full of trajectory epoch times.

        Parameters
        ----------
        batch_size : int
        epoch : int
        indices : ndarray or torch.Tensor or None
            Selected indices for iteration.
            If None, whole trajectory is selected.
        shuffle : bool

        Returns
        -------
        data_map : dict of torch.Tensor
        """

        for _ in range(epoch):
            indices = self._get_indices(indices, shuffle)
            while self._next_id <= len(indices) - batch_size:
                yield self._next_batch(batch_size, indices)
            self._next_id = 0

    def iterate_step(self, batch_size, step=1, indices=None, shuffle=True):
        indices = self._get_indices(indices, shuffle)
        for _ in range(step):
            self._next_id = self._next_id % len(indices)
            yield self._next_batch(batch_size, indices)

    def random_batch_once(self, batch_size, indices=None, return_indices=False):
        """
        Providing a batch which is randomly sampled from trajectory.

        Parameters
        ----------
        batch_size : int
        indices : ndarray or torch.Tensor or None
            Selected indices for iteration.
            If None, whole trajectory is selected.
        return_indices : bool
            If True, indices are also returned.

        Returns
        -------
        data_map : dict of torch.Tensor
        """
        if indices is not None:
            indices = self._get_indices(indices, shuffle=True)
        else:
            indices = torch.randint(0, self.num_step - 1, size=(batch_size, ))

        if self.ddp:
            indices = indices[self.rank:len(indices):self.world_size]

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][indices].to(
                get_device())
        if return_indices:
            return data_map, indices
        else:
            return data_map

    def prioritized_random_batch_once(self, batch_size, return_indices=False, mode='proportional', alpha=0.6, init_beta=0.4, beta_step=0.00025/4):
        if hasattr(self, 'pri_beta') == False:
            self.pri_beta = init_beta
        elif self.pri_beta >= 1.0:
            self.pri_beta = 1.0
        else:
            self.pri_beta += beta_step

        pris = self.data_map['pris'].cpu().numpy()

        if mode == 'rank_based':
            index = np.argsort(-pris)
            pris = (index.astype(np.float32)+1) ** -1
            pris = pris ** alpha

        is_weights = (len(pris) * (pris/pris.sum())) ** -self.pri_beta
        is_weights /= np.max(is_weights)
        pris *= is_weights
        pris = torch.tensor(pris)
        indices = torch.utils.data.sampler.WeightedRandomSampler(
            pris, batch_size, replacement=True)
        indices = [index for index in indices]

        if self.ddp:
            indices = indices[self.rank:len(indices):self.world_size]

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][indices].to(get_device())
        if return_indices:
            return data_map, indices
        else:
            return data_map

    def prioritized_random_batch_rnn_once(self, batch_size, seq_length, return_indices=False, init_beta=0.4, beta_step=0.00025/4):
        if hasattr(self, 'pri_beta') == False:
            self.pri_beta = init_beta
        elif self.pri_beta >= 1.0:
            self.pri_beta = 1.0
        else:
            self.pri_beta += beta_step

        seq_pris = self.data_map['seq_pris'].clone().detach()

        start_indices = torch.utils.data.sampler.WeightedRandomSampler(
            seq_pris, batch_size)  # , replacement=True)
        start_indices = [idx for idx in start_indices]

        seqs = []
        length = []
        for start in start_indices:
            data_map = dict()
            for key in self.data_map:
                data_map[key] = self.data_map[key][start: start+seq_length]
            seqs.append(data_map)

        batch = dict()
        keys = seqs[0].keys()
        for key in keys:
            batch[key] = torch.stack([seq[key] for seq in seqs], dim=0)
            # (batch_size, seq_length, *) -> (seq_length, batch_size, *)
            batch[key] = batch[key].transpose(0, 1).to(get_device())

        if return_indices:
            return batch, start_indices
        else:
            return batch

    def random_batch(self, batch_size, epoch=1, indices=None, return_indices=False):
        """
        Providing batches which is randomly sampled from trajectory.

        Parameters
        ----------
        batch_size : int
        epoch : int
        indices : ndarray or torch.Tensor or None
            Selected indices for iteration.
            If None, whole trajectory is selected.
        return_indices : bool
            If True, indices are also returned.

        Returns
        -------
        data_map : dict of torch.Tensor
        """
        for _ in range(epoch):
            if return_indices:
                batch, indices = self.random_batch_once(
                    batch_size, indices, return_indices)
                yield batch, indices
            else:
                batch = self.random_batch_once(
                    batch_size, indices, return_indices)
                yield batch

    def random_batch_rnn(self, batch_size, seq_length=None, epoch=1):
        """
        Providing sequences of batch which is randomly sampled from trajectory.
        batch shape is (seq_length, batch_size, * )

        Parameters
        ----------
        batch_size : int
        seq_length : int
            Length of sequence of batch. 
            If seq_length None, max episode length is selected.
        epoch : int

        Returns
        -------
        batch : dict of torch.Tensor
        """

        if seq_length is None:
            seq_length = max([self._epis_index[i+1] - self._epis_index[i]
                              for i in range(len(self._epis_index)-1)])

        for _ in range(epoch):
            seqs = []
            lengths = []
            indices = np.random.randint(
                0, len(self._epis_index)-1, (batch_size,))

            if self.ddp:
                indices = indices[self.rank:len(indices):self.world_size]

            for idx in indices:
                length = min(
                    self._epis_index[idx+1] - self._epis_index[idx], seq_length)
                start = np.random.randint(
                    self._epis_index[idx], self._epis_index[idx+1] - length + 1)
                data_map = dict()
                for key in self.data_map:
                    if self._epis_index[-1] - self._epis_index[idx] < seq_length:
                        pad = torch.zeros_like(self.data_map[key][:seq_length-length],
                                               dtype=torch.float, device=get_device())
                        data_map[key] = torch.cat(
                            [self.data_map[key][start: start+length], pad])
                    else:
                        data_map[key] = self.data_map[key][start: start+seq_length]
                lengths.append(length)
                seqs.append(data_map)

            batch = dict()
            keys = seqs[0].keys()
            for key in keys:
                batch[key] = torch.stack([seq[key] for seq in seqs])
                # (batch_size, seq_length, *) -> (seq_length, batch_size, *)
                batch[key] = batch[key].transpose(0, 1).to(get_device())
            out_masks = torch.ones(
                (seq_length, batch_size), dtype=torch.float, device=get_device())
            for i in range(batch_size):
                out_masks[lengths[i]:, i] = 0
            batch['out_masks'] = out_masks
            yield batch

    def prioritized_random_batch(self, batch_size, epoch=1, return_indices=False):
        for _ in range(epoch):
            if return_indices:
                batch, indices = self.prioritized_random_batch_once(
                    batch_size, return_indices)
                yield batch, indices
            else:
                batch = self.prioritized_random_batch_once(
                    batch_size, return_indices)
                yield batch

    def prioritized_random_batch_rnn(self, batch_size, seq_length, epoch=1, return_indices=False):
        """
        Providing sequences of batch which is prioritized randomly sampled from trajectory.
        batch shape is (seq_length, batch_size, * )

        Parameters
        ----------
        batch_size : int
        seq_length : int
            Length of sequence of batch. 
            If seq_length None, max episode length is selected.
        epoch : int
        return_indices : bool
            If True, indices are also returned.

        Returns
        -------
        batch : dict of torch.Tensor
        """
        for _ in range(epoch):
            if return_indices:
                batch, start_indices = self.prioritized_random_batch_rnn_once(
                    batch_size, seq_length, return_indices)
                yield batch, start_indices
            else:
                batch = self.prioritized_random_batch_once(
                    batch_size, seq_length, return_indices)
                yield batch

    def full_batch(self, epoch=1, return_indices=False):
        """
        Providing whole trajectory as batch.

        Parameters
        ----------
        epoch : int
        return_indices : bool
            If True, indices are also returned.

        Returns
        -------
        data_map : dict of torch.Tensor
        """
        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key].to(get_device())
        for _ in range(epoch):
            if return_indices:
                yield data_map, torch.arange(self.num_step)
            else:
                yield data_map

    def iterate_epi(self, shuffle=True):
        """
        Iterating episodes.

        Parameters
        ----------
        shuffle : bool

        Returns
        -------
        epis : dict of torch.Tensor
        """
        epis = []
        for i in range(len(self._epis_index) - 1):
            data_map = dict()
            for key in self.data_map:
                data_map[key] = self.data_map[key][self._epis_index[i]:self._epis_index[i+1]]
            epis.append(data_map)
        if shuffle:
            indices = np.random.permutation(range(len(epis)))
        else:
            indices = range(len(epis))
        for idx in indices:
            yield epis[idx]

    def iterate_rnn(self, batch_size, num_epi_per_seq=1, epoch=1):
        """
        Iterating batches for rnn.
        batch shape is (max_seq, batch_size, * )

        Parameters
        ----------
        batch_size : int
        num_epi_per_seq : int
            Number of episodes in one sequence for rnn.
        epoch : int

        Returns
        -------
        batch : dict of torch.Tensor
        """
        assert batch_size * num_epi_per_seq <= self.num_epi
        for _ in range(epoch):
            epi_count = 0
            all_batch = []
            seq = []
            for epi in self.iterate_epi(shuffle=True):
                seq.append(epi)
                epi_count += 1
                if epi_count >= num_epi_per_seq:
                    _seq = dict()
                    for key in seq[0].keys():
                        _seq[key] = torch.cat([s[key] for s in seq])
                    all_batch.append(_seq)
                    seq = []
                    epi_count = 0
            num_batch = len(all_batch)
            idx = 0
            while idx <= num_batch - batch_size:
                cur_batch_size = min(batch_size, num_batch - idx)
                batch = all_batch[idx:idx+cur_batch_size]
                idx += cur_batch_size

                lengths = [list(b.values())[0].size(0) for b in batch]
                max_length = max(lengths)
                out_masks = torch.ones(
                    (max_length, cur_batch_size), dtype=torch.float, device=get_device())
                time_slice = list(functools.reduce(
                    lambda x, y: x+y, [list(range(l, max_length)) for l in lengths]))
                batch_idx = list(functools.reduce(
                    lambda x, y: x+y, [(max_length - l) * [i] for i, l in enumerate(lengths)]))
                out_masks[time_slice, batch_idx] = 0

                _batch = dict()
                keys = batch[0].keys()
                for key in keys:
                    _batch[key] = pad_sequence(
                        [b[key] for b in batch]).to(get_device())
                _batch['out_masks'] = out_masks.to(get_device())
                yield _batch

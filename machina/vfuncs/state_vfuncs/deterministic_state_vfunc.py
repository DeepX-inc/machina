"""
Deterministic State Value function
"""

import torch
import torch.nn as nn

from machina.pds import DeterministicPd
from machina.vfuncs.state_vfuncs.base import BaseSVfunc
from machina.utils import get_device


class DeterministicSVfunc(BaseSVfunc):
    """
    Deterministic version of State Action Value Function.

    Parameters
    ----------
    ob_space : gym.Space
    net : torch.nn.Module
    rnn : bool
    data_parallel : bool
        If True, network computation is executed in parallel.
    parallel_dim : int
        Splitted dimension in data parallel.
    """

    def __init__(self, ob_space, net, rnn=False, data_parallel=False, parallel_dim=0):
        super().__init__(ob_space, net, rnn, data_parallel, parallel_dim)
        self.pd = DeterministicPd()
        self.to(get_device())

    def forward(self, obs, hs=None, h_masks=None):
        """
        Calculating values.
        """
        obs = self._check_obs_shape(obs)

        if self.rnn:
            time_seq, batch_size, *_ = obs.shape
            if hs is None:
                if self.hs is None:
                    self.hs = self.net.init_hs(batch_size)
                if self.dp_run:
                    self.hs = (self.hs[0].unsqueeze(
                        0), self.hs[1].unsqueeze(0))
                hs = self.hs

            if h_masks is None:
                h_masks = hs[0].new(time_seq, batch_size, 1).zero_()
            h_masks = h_masks.reshape(time_seq, batch_size, 1)

            if self.dp_run:
                vs, hs = self.dp_net(obs, hs, h_masks)
            else:
                vs, hs = self.net(obs, hs, h_masks)
            self.hs = hs
        else:
            if self.dp_run:
                vs = self.dp_net(obs)
            else:
                vs = self.net(obs)
        return vs.squeeze(-1), dict(mean=vs.squeeze(-1), hs=hs)


class NormalizedDeterministicSVfunc(DeterministicSVfunc):
    def __init__(self, ob_space, net):
        super().__init__(self, ob_space, net)
        self.x_mean = torch.zeros(1)
        self.x_std = torch.ones(1)
        self.to(get_device())

        self.normalized = True

    def forward(self, obs, hs=None, h_masks=None):
        vs, info = super().forward(obs, hs, h_masks)
        return vs * self.x_std + self.x_mean, info

    def set_mean(self, mean):
        self.x_mean.data.copy_(mean)

    def set_std(self, std):
        self.x_std.data.copy_(std)

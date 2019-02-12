import numpy as np
import torch
import torch.nn as nn
from machina.pols import BasePol
from machina.pds.multi_categorical_pd import MultiCategoricalPd
from machina.utils import get_device


class MultiCategoricalPol(BasePol):
    """
    Policy with Categorical distribution.

    Parameters
    ----------
    ob_space : gym.Space
        observation's space
    ac_space : gym.Space
        action's space.
        This should be gym.spaces.MultiDiscrete
    net : torch.nn.Module
    rnn : bool
    normalize_ac : bool
        If True, the output of network is spreaded for ac_space.
        In this situation the output of network is expected to be in -1~1.
    data_parallel : bool
        If True, network computation is executed in parallel.
    parallel_dim : int
        Splitted dimension in data parallel.
    """

    def __init__(self, ob_space, ac_space, net, rnn=False, normalize_ac=True, data_parallel=False, parallel_dim=0):
        BasePol.__init__(self, ob_space, ac_space, net, rnn,
                         normalize_ac, data_parallel, parallel_dim)
        self.pd = MultiCategoricalPd()
        self.to(get_device())

    def forward(self, obs, hs=None, h_masks=None):
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
                pis, hs = self.dp_net(obs, hs, h_masks)
            else:
                pis, hs = self.net(obs, hs, h_masks)
            self.hs = hs
        else:
            if self.dp_run:
                pis = self.dp_net(obs)
            else:
                pis = self.net(obs)
        ac = self.pd.sample(dict(pis=pis))
        ac_real = self.convert_ac_for_real(ac.detach().cpu().numpy())
        return ac_real, ac, dict(pis=pis, hs=hs)

    def deterministic_ac_real(self, obs, hs=None, h_masks=None):
        """
        action for deployment
        """
        obs = self._check_obs_shape(obs)

        if self.rnn:
            time_seq, batch_size, *_ = obs.shape
            if hs is None:
                if self.hs is None:
                    self.hs = self.net.init_hs(batch_size)
                hs = self.hs

            if h_masks is None:
                h_masks = hs[0].new(time_seq, batch_size, 1).zero_()
            h_masks = h_masks.reshape(time_seq, batch_size, 1)

            pis, hs = self.net(obs, hs, h_masks)
            self.hs = hs
        else:
            pis = self.net(obs)
        _, ac = torch.max(pis, dim=-1)
        ac_real = self.convert_ac_for_real(ac.detach().cpu().numpy())
        return ac_real, ac, dict(pis=pis, hs=hs)

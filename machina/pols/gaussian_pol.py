import numpy as np
import torch
import torch.nn as nn

from machina.pols import BasePol
from machina.pds.gaussian_pd import GaussianPd
from machina.utils import get_device


class GaussianPol(BasePol):
    """
    Policy with Gaussian distribution.

    Parameters
    ----------
    observation_space : gym.Space
        observation's space
    action_space : gym.Space
        action's space
        This should be gym.spaces.Box
    net : torch.nn.Module
    rnn : bool
    normalize_ac : bool
        If True, the output of network is spreaded for action_space.
        In this situation the output of network is expected to be in -1~1.
    """

    def __init__(self, observation_space, action_space, net, rnn=False,
                 normalize_ac=True):
        BasePol.__init__(self, observation_space, action_space, net, rnn,
                         normalize_ac)
        self.pd = GaussianPd()
        self.to(get_device())

    def forward(self, obs, hs=None, h_masks=None):
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

            mean, log_std, hs = self.net(obs, hs, h_masks)
            self.hs = hs
        else:
            mean, log_std = self.net(obs)
        log_std = log_std.expand_as(mean)
        ac = self.pd.sample(dict(mean=mean, log_std=log_std))
        ac_real = self.convert_ac_for_real(ac.detach().cpu().numpy())
        return ac_real, ac, dict(mean=mean, log_std=log_std, hs=hs)

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

            mean, log_std, hs = self.net(obs, hs, h_masks)
            self.hs = hs
        else:
            mean, log_std = self.net(obs)
        mean_real = self.convert_ac_for_real(mean.detach().cpu().numpy())
        return mean_real, mean, dict(mean=mean, log_std=log_std, hs=hs)

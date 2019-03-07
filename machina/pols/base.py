import gym
import numpy as np
import torch
import torch.nn as nn


class BasePol(nn.Module):
    """
    Base class of Policy.

    Parameters
    ----------
    ob_space : gym.Space
        observation's space
    ac_space : gym.Space
        action's space
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
        nn.Module.__init__(self)
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.net = net

        self.rnn = rnn
        self.hs = None

        self.normalize_ac = normalize_ac
        self.data_parallel = data_parallel
        if data_parallel:
            self.dp_net = nn.DataParallel(self.net, dim=parallel_dim)
        self.dp_run = False

        self.discrete = isinstance(ac_space, gym.spaces.MultiDiscrete) or isinstance(
            ac_space, gym.spaces.Discrete)
        self.multi = isinstance(ac_space, gym.spaces.MultiDiscrete)

        if not self.discrete:
            self.a_i_shape = ac_space.shape
        else:
            if isinstance(ac_space, gym.spaces.MultiDiscrete):
                nvec = ac_space.nvec
                assert any([nvec[0] == nv for nv in nvec])
                self.a_i_shape = (len(nvec), nvec[0])
            elif isinstance(ac_space, gym.spaces.Discrete):
                self.a_i_shape = (ac_space.n, )

    def convert_ac_for_real(self, x):
        """
        Converting action which is output of network for real world value.
        """
        if not self.discrete:
            lb, ub = self.ac_space.low, self.ac_space.high
            if self.normalize_ac:
                x = lb + (x + 1.) * 0.5 * (ub - lb)
                x = np.clip(x, lb, ub)
            else:
                x = np.clip(x, lb, ub)
        return x

    def reset(self):
        """
        reset for rnn's hidden state.
        """
        if self.rnn:
            self.hs = None

    def _check_obs_shape(self, obs):
        """
        Reshape input appropriately.
        """
        if self.rnn:
            additional_shape = 2
        else:
            additional_shape = 1
        if len(obs.shape) < additional_shape + len(self.ob_space.shape):
            for _ in range(additional_shape + len(self.ob_space.shape) - len(obs.shape)):
                obs = obs.unsqueeze(0)
        return obs

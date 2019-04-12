import copy

import gym
import numpy as np
import torch
import torch.nn as nn

from machina.utils import get_device


class BasePol(nn.Module):
    """
    Base class of Policy.

    Parameters
    ----------
    observation_space : gym.Space
        observation's space
    action_space : gym.Space
        action's space
    net : torch.nn.Module
    rnn : bool
    normalize_ac : bool
        If True, the output of network is spreaded for action_space.
        In this situation the output of network is expected to be in -1~1.
    data_parallel : bool or str
        If True, network computation is executed in parallel.
        If data_parallel is ddp, network computation is executed in distributed parallel.
    parallel_dim : int
        Splitted dimension in data parallel.
    """

    def __init__(self, observation_space, action_space, net, rnn=False, normalize_ac=True, data_parallel=False, parallel_dim=0):
        nn.Module.__init__(self)
        self.observation_space = observation_space
        self.action_space = action_space
        self.net = net

        self.rnn = rnn
        self.hs = None

        self.normalize_ac = normalize_ac
        self.data_parallel = data_parallel
        if data_parallel:
            if data_parallel is True:
                self.dp_net = nn.DataParallel(self.net, dim=parallel_dim)
            elif data_parallel == 'ddp':
                self.net.to(get_device())
                self.dp_net = nn.parallel.DistributedDataParallel(
                    self.net, device_ids=[get_device()], dim=parallel_dim)
            else:
                raise ValueError(
                    'Bool and str(ddp) are allowed to be data_parallel.')
        self.dp_run = False

        self.discrete = isinstance(action_space, gym.spaces.MultiDiscrete) or isinstance(
            action_space, gym.spaces.Discrete)
        self.multi = isinstance(action_space, gym.spaces.MultiDiscrete)

        if not self.discrete:
            self.a_i_shape = action_space.shape
        else:
            if isinstance(action_space, gym.spaces.MultiDiscrete):
                nvec = action_space.nvec
                assert any([nvec[0] == nv for nv in nvec])
                self.a_i_shape = (len(nvec), nvec[0])
            elif isinstance(action_space, gym.spaces.Discrete):
                self.a_i_shape = (action_space.n, )

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'dp_net' in state['_modules']:
            _modules = copy.deepcopy(state['_modules'])
            del _modules['dp_net']
            state['_modules'] = _modules
        return state

    def __setstate__(self, state):
        if 'dp_net' in state:
            state.pop('dp_net')
        self.__dict__.update(state)

    def convert_ac_for_real(self, x):
        """
        Converting action which is output of network for real world value.
        """
        if not self.discrete:
            lb, ub = self.action_space.low, self.action_space.high
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
        if len(obs.shape) < additional_shape + len(self.observation_space.shape):
            for _ in range(additional_shape + len(self.observation_space.shape) - len(obs.shape)):
                obs = obs.unsqueeze(0)
        return obs

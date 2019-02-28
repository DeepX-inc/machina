import numpy as np
import random
import torch
import torch.nn as nn

from machina.pols import BasePol
from machina.pds.gaussian_pd import GaussianPd
from machina.utils import get_device
from torch.distributions import MultivariateNormal


class ArgmaxQfPol(BasePol):
    """
    Policy with Continuous Qfunction.

    Parameters
    ----------
    ob_space : gym.Space
        observation's space
    ac_space : gym.Space
        action's space
        This should be gym.spaces.Box
    net : torch.nn.Module
    rnn : bool
    normalize_ac : bool
        If True, the output of network is spreaded for ac_space.
        In this situation the output of network is expected to be in -1~1.
    data_parallel : bool
        If True, network computation is executed in parallel.
    parallel_dim : int
        Splitted dimension in data parallel.
    eps : float
        Probability of random action
    """

    def __init__(self, ob_space, ac_space, qfunc, rnn=False, normalize_ac=True, data_parallel=False, parallel_dim=0, eps=0.2, env=None):
        BasePol.__init__(self, ob_space, ac_space, None, rnn,
                         normalize_ac, data_parallel, parallel_dim)
        self.qfunc = qfunc
        self.eps = eps
        self.env = env
        self.to(get_device())

    def forward(self, obs):
        prob = random.uniform(0., 1.)
        if prob <= self.eps:
            ac_real = ac = torch.tensor(self.env.action_space.sample(), dtype=torch.float, device=get_device())
        else:
            q, ac = self.qfunc.max(obs)
            ac_real = self.convert_ac_for_real(ac.detach().cpu().numpy())
        return ac_real, ac, dict()

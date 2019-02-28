import numpy as np
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
    """

    def __init__(self, ob_space, ac_space, qfunc, rnn=False, normalize_ac=True, data_parallel=False, parallel_dim=0):
        BasePol.__init__(self, ob_space, ac_space, None, rnn,
                         normalize_ac, data_parallel, parallel_dim)
        self.qfunc = qfunc
        self.to(get_device())

    def forward(self, obs):
        q, ac = self.qfunc.max(obs)
        ac_real = self.convert_ac_for_real(ac.detach().cpu().numpy())
        return ac_real, ac, dict()

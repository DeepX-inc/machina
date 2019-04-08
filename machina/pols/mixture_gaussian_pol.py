import numpy as np
import torch
from torch.distributions import Categorical
from machina.pols import BasePol
from machina.pds.mixture_gaussian_pd import MixtureGaussianPd
from machina.utils import get_device


class MixtureGaussianPol(BasePol):
    def __init__(self, observation_space, action_space, net, normalize_ac=True):
        BasePol.__init__(self, observation_space, action_space, normalize_ac)
        self.net = net
        self.pd = MixtureGaussianPd()
        self.to(get_device())

    def forward(self, obs):
        pi, mean, log_std = self.net(obs)
        log_std = log_std.expand_as(mean)
        ac = self.pd.sample(dict(pi=pi, mean=mean, log_std=log_std))
        ac_real = self.convert_ac_for_real(ac.detach().cpu().numpy())
        return ac_real, ac, dict(pi=pi, mean=mean, log_std=log_std)

    def deterministic_ac_real(self, obs):
        """
        action for deployment
        """
        pi, mean, _ = self.net(obs)
        _, i = torch.max(pi, 1)
        onehot = torch.zeros_like(mean)
        onehot = onehot.scatter_(-1, i.unsqueeze(-1), 1)
        mean_real = self.convert_ac_for_real(
            torch.sum(mean * onehot.unsqueeze(-1), 1).detach().cpu().numpy())
        return mean_real

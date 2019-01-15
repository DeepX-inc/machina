
import numpy as np
import torch
from torch.distributions import Normal, kl_divergence

from machina.pds.base import BasePd


class GaussianPd(BasePd):
    """
    Gaussian probablistic distribution.
    """

    def sample(self, params, sample_shape=torch.Size()):
        mean, log_std = params['mean'], params['log_std']
        std = torch.exp(log_std)
        ac = Normal(loc=mean, scale=std).rsample(sample_shape)
        return ac

    def llh(self, x, params):
        mean, log_std = params['mean'], params['log_std']
        std = torch.exp(log_std)
        return torch.sum(Normal(loc=mean, scale=std).log_prob(x), dim=-1)

    def kl_pq(self, p_params, q_params):
        p_mean, p_log_std = p_params['mean'], p_params['log_std']
        q_mean, q_log_std = q_params['mean'], q_params['log_std']
        p_std = torch.exp(p_log_std)
        q_std = torch.exp(q_log_std)
        return torch.sum(kl_divergence(Normal(loc=p_mean, scale=p_std), Normal(loc=q_mean, scale=q_std)), dim=-1)

    def ent(self, params):
        mean = params['mean']
        log_std = params['log_std']
        std = torch.exp(log_std)
        return torch.sum(Normal(loc=mean, scale=std).entropy(), dim=-1)

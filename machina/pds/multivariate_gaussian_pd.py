
import numpy as np
import torch
from torch.distributions import MultivatiateNormal, kl_divergence

from machina.pds.base import BasePd


class MultivariateGaussianPd(BasePd):
    """
    Multivariate Gaussian probablistic distribution.
    """

    def sample(self, params, sample_shape=torch.Size()):
        mean, stddev = params['mean'], params['stddev']
        ac = MultivariateNormal(loc=mean, scale=std).rsample(sample_shape)
        return ac

    def llh(self, x, params):
        mean, stddev = params['mean'], params['stddev']
        return torch.sum(MultivariateNormal(loc=mean, scale=std).log_prob(x), dim=-1)

    def kl_pq(self, p_params, q_params):
        p_mean, p_stddev = p_params['mean'], p_params['stddev']
        q_mean, q_stddev = q_params['mean'], q_params['stddev']
        return torch.sum(kl_divergence(MultivariateNormal(loc=p_mean, scale=p_std), MultivariateNormal(loc=q_mean, scale=q_std)), dim=-1)

    def ent(self, params):
        mean = params['mean']
        stddev = params['stddev']
        return torch.sum(MultivariateNormal(loc=mean, scale=std).entropy(), dim=-1)

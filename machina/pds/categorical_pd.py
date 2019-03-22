"""
Categorical
"""

import numpy as np
import torch
from torch.distributions import Categorical, kl_divergence

from machina.pds.base import BasePd


class CategoricalPd(BasePd):
    """
    Categorical probablistic distribution.
    """

    def sample(self, params, sample_shape=torch.Size()):
        pi = params['pi']
        pi_sampled = Categorical(probs=pi).sample(sample_shape)
        return pi_sampled

    def llh(self, x, params):
        pi = params['pi']
        return Categorical(pi).log_prob(x)

    def kl_pq(self, p_params, q_params):
        p_pi = p_params['pi']
        q_pi = q_params['pi']
        return kl_divergence(Categorical(p_pi), Categorical(q_pi))

    def ent(self, params):
        pi = params['pi']
        return Categorical(pi).entropy()

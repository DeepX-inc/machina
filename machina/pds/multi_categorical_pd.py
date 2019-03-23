
import numpy as np
import torch
from torch.distributions import Categorical, kl_divergence

from machina.pds.base import BasePd


class MultiCategoricalPd(BasePd):
    """
    Multi Categorical probablistic distribution
    """

    def sample(self, params, sample_shape=torch.Size()):
        pis = params['pis']
        pis_sampled = []
        for pi in torch.chunk(pis, pis.size(-2), -2):
            pi_sampled = Categorical(probs=pi).sample()
            pis_sampled.append(pi_sampled)
        return torch.cat(pis_sampled, dim=-1)

    def llh(self, xs, params):
        pis = params['pis']
        llhs = []
        for x, pi in zip(torch.chunk(xs, xs.size(-1), -1), torch.chunk(pis, pis.size(-2), -2)):
            x = x.squeeze(-1)
            pi = pi.squeeze(-2)
            llhs.append(Categorical(pi).log_prob(x))
        return sum(llhs)

    def kl_pq(self, p_params, q_params):
        p_pis = p_params['pis']
        q_pis = q_params['pis']
        kls = []
        for p_pi, q_pi in zip(torch.chunk(p_pis, p_pis.size(-2), -2), torch.chunk(q_pis, q_pis.size(-2), -2)):
            kls.append(kl_divergence(Categorical(p_pi), Categorical(q_pi)))
        return sum(kls)

    def ent(self, params):
        pis = params['pis']
        ents = []
        for pi in torch.chunk(pis, pis.size(-2), -2):
            ents.append(torch.sum(Categorical(pi).entropy(), dim=-1))
        return sum(ents)

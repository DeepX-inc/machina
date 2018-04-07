import numpy as np
import torch
from torch.autograd import Variable

from machina.pds.base import BasePd

class GaussianPd(BasePd):
    def __init__(self, ob_space, ac_space):
        BasePd.__init__(self, ob_space, ac_space)

    def sample(self, params):
        mean, log_std = params['mean'], params['log_std']
        ac = mean + Variable(mean.data.new(*mean.shape).normal_()) * torch.exp(log_std)
        return ac

    def llh(self, x, params):
        mean, log_std = params['mean'], params['log_std']
        std = torch.exp(log_std)
        return - 0.5 * torch.sum(((x - mean) / std)**2, dim=-1) \
                - 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                - torch.sum(log_std, dim=-1)

    def kl_pq(self, p_params, q_params):
        p_mean, p_log_std = p_params['mean'], p_params['log_std']
        q_mean, q_log_std = q_params['mean'], q_params['log_std']
        p_std = torch.exp(p_log_std)
        q_std = torch.exp(q_log_std)
        return torch.sum(((p_mean - q_mean)**2 + p_std**2 - q_std**2) / (2 * q_std**2 + 1e-8) + q_log_std - p_log_std, dim=-1)


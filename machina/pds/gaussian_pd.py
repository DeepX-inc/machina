import torch
import numpy as np
from .base import BasePd

class GaussianPd(BasePd):
    def __init__(self, ob_space, ac_space):
        BasePd.__init__(self, ob_space, ac_space)

    def llh(self, x, mean, log_std):
        std = torch.exp(log_std)
        return - 0.5 * torch.sum(((x - mean) / std)**2, dim=-1) \
                - 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                - torch.sum(log_std, dim=-1)

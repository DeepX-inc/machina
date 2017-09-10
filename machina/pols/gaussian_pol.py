import numpy as np
import torch
from torch.autograd import Variable
from .base import BasePol
from ..pds.gaussian_pd import GaussianPd

class GaussianPol(BasePol):
    def __init__(self, ob_space, ac_space, net, normalize_ac=True):
        BasePol.__init__(self, ob_space, ac_space, normalize_ac)
        self.net = net
        self.pd = GaussianPd(ob_space, ac_space)

    def forward(self, obs):
        mean, log_std = self.net(obs)
        ac = mean + Variable(torch.randn(self.ac_space.shape)) * torch.exp(log_std)
        ac_real = ac.data.cpu().numpy()
        if self.normalize_ac:
            lb, ub = self.ac_space.low, self.ac_space.high
            ac_real = lb + (ac_real + 1.) * 0.5 * (ub - lb)
            ac_real = np.clip(ac_real, lb, ub)
        return ac_real, ac, dict(mean=mean, log_std=log_std)

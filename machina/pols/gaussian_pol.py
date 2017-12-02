import numpy as np
import torch
from machina.pols import BasePol
from machina.pds.gaussian_pd import GaussianPd
from machina.utils import Variable, get_gpu, torch2torch

class GaussianPol(BasePol):
    def __init__(self, ob_space, ac_space, net, normalize_ac=True):
        BasePol.__init__(self, ob_space, ac_space, normalize_ac)
        self.net = net
        self.pd = GaussianPd(ob_space, ac_space)
        gpu_id = get_gpu()
        if gpu_id != -1:
            self.cuda(gpu_id)

    def forward(self, obs):
        mean, log_std = self.net(obs)
        log_std = log_std.expand_as(mean)
        ac = mean + Variable(torch2torch(torch.randn(mean.size()))) * torch.exp(log_std)
        ac_real = ac.data.cpu().numpy()
        lb, ub = self.ac_space.low, self.ac_space.high
        if self.normalize_ac:
            ac_real = lb + (ac_real + 1.) * 0.5 * (ub - lb)
            ac_real = np.clip(ac_real, lb, ub)
        else:
            ac_real = np.clip(ac_real, lb, ub)
        return ac_real, ac, dict(mean=mean, log_std=log_std)

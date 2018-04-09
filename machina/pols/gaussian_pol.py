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
        ac = self.pd.sample(dict(mean=mean, log_std=log_std))
        ac_real = self.convert_ac_for_real(ac.data.cpu().numpy())
        return ac_real, ac, dict(mean=mean, log_std=log_std)

    def deterministic_ac_real(self, obs):
        """
        action for deployment
        """
        mean, _ = self.net(obs)
        mean_real = self.convert_ac_for_real(mean.data.cpu().numpy())
        return mean_real



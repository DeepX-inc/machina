import numpy as np
import torch
from torch.distributions import Categorical
from machina.pols import BasePol
from machina.pds.mixture_gaussian_pd import MixtureGaussianPd
from machina.utils import Variable, get_gpu, torch2torch

class MixtureGaussianPol(BasePol):
    def __init__(self, ob_space, ac_space, net, normalize_ac=True):
        BasePol.__init__(self, ob_space, ac_space, normalize_ac)
        self.net = net
        self.pd = MixtureGaussianPd(ob_space, ac_space)
        gpu_id = get_gpu()
        if gpu_id != -1:
            self.cuda(gpu_id)

    def forward(self, obs):
        pi, mean, log_std = self.net(obs)
        log_std = log_std.expand_as(mean)
        ac = self.pd.sample(dict(pi=pi, mean=mean, log_std=log_std))
        ac_real = self.convert_ac_for_real(ac.data.cpu().numpy())
        return ac_real, ac, dict(pi=pi, mean=mean, log_std=log_std)

    def deterministic_ac_real(self, obs):
        """
        action for deployment
        """
        pi, mean, _ = self.net(obs)
        _, i = torch.max(pi, 1)
        onehot = mean.new(*mean.shape).zero_()
        onehot = onehot.scatter_(-1, i.unsqueeze(-1), 1)
        mean_real = self.convert_ac_for_real(torch.sum(mean * onehot.unsqueeze(-1), 1).data.cpu().numpy())
        return mean_real


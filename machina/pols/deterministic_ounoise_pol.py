import numpy as np
import torch
from .base import BasePol
from ..utils import Variable, gpu_id

class ActionNoise(object):
    def reset(self):
        pass


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mu, dtype = np.float32)


class DeterministicOUNoisePol(BasePol):
    def __init__(self, ob_space, ac_space, net, noise, normalize_ac=True):
        BasePol.__init__(self, ob_space, ac_space, normalize_ac)
        self.net = net
        if gpu_id != -1:
            self.cuda(gpu_id)
        self.noise = noise

    def reset(self):
        self.noise.reset()

    def forward(self, obs):
        mean = self.net(obs)
        action_noise = self.noise()
        ac = mean + Variable(torch.from_numpy(action_noise))
        ac_real = ac.data.cpu().numpy()
        lb, ub = self.ac_space.low, self.ac_space.high
        if self.normalize_ac:
            ac_real = lb + (ac_real + 1.) * 0.5 * (ub - lb)
            ac_real = np.clip(ac_real, lb, ub)
        else:
            ac_real = np.clip(ac_real, lb, ub)
        return ac_real, ac, dict(mean=mean)


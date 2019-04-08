"""
This is implementation of Ornstein-Uhlenbeck process.
"""

import numpy as np
import torch

from machina.utils import get_device
from machina.noise.base import BaseActionNoise


class OUActionNoise(BaseActionNoise):
    """
    noise produced by Ornstein-Uhlenbeck process.
    """

    def __init__(self, action_space, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        BaseActionNoise.__init__(self, action_space)
        self.mu = np.zeros(self.action_space.shape[0])
        self.theta = theta
        self.sigma = sigma * np.ones_like(self.mu)
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self, device='cpu'):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return torch.tensor(x, dtype=torch.float, device=device)

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mu, dtype=np.float32)

import numpy as np
import torch

from machina.pds import DeterministicPd
from machina.pols import BasePol
from machina.utils import get_device


class DeterministicActionNoisePol(BasePol):
    """
    Policy with deterministic distribution.

    Parameters
    ----------
    observation_space : gym.Space
        observation's space
    action_space : gym.Space
        action's space.
        This should be gym.spaces.Box
    net : torch.nn.Module
    noise : Noise
    rnn : bool
    normalize_ac : bool
        If True, the output of network is spreaded for action_space.
        In this situation the output of network is expected to be in -1~1.
    """

    def __init__(self, observation_space, action_space, net, noise=None, rnn=False, normalize_ac=True):
        if rnn:
            raise ValueError(
                'rnn with DeterministicActionNoisePol is not supported now')
        BasePol.__init__(self, observation_space, action_space,
                         net, rnn=rnn, normalize_ac=normalize_ac)
        self.noise = noise
        self.pd = DeterministicPd()
        self.to(get_device())

    def reset(self):
        super(DeterministicActionNoisePol, self).reset()
        if self.noise is not None:
            self.noise.reset()

    def forward(self, obs, no_noise=False):
        obs = self._check_obs_shape(obs)

        mean = self.net(obs)
        ac = mean

        if self.noise is not None and not no_noise:
            action_noise = self.noise(device=ac.device)
            ac = ac + action_noise

        ac_real = self.convert_ac_for_real(ac.detach().cpu().numpy())
        return ac_real, ac, dict(mean=mean)

    def deterministic_ac_real(self, obs):
        """
        action for deployment
        """
        obs = self._check_obs_shape(obs)

        mean = self.net(obs)
        mean_real = self.convert_ac_for_real(mean.detach().cpu().numpy())
        return mean_real, mean, dict(mean=mean)

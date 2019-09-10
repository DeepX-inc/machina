import random
import torch

from machina.pols import BasePol
from machina.utils import get_device


class ArgmaxQfPol(BasePol):
    """
    Policy with Continuous Qfunction.

    Parameters
    ----------
    observation_space : gym.Space
        observation's space
    action_space : gym.Space
        action's space
        This should be gym.spaces.Box
    qfunc : SAVfunc
    rnn : bool
    normalize_ac : bool
        If True, the output of network is spreaded for action_space.
        In this situation the output of network is expected to be in -1~1.
    eps : float
        Probability of random action
    """

    def __init__(self, observation_space, action_space, qfunc, rnn=False, normalize_ac=True, eps=0.2):
        BasePol.__init__(self, observation_space,
                         action_space, None, rnn, normalize_ac)
        self.qfunc = qfunc
        self.eps = eps
        self.a_i_shape = (1, )
        self.to(get_device())

    def forward(self, obs):
        prob = random.uniform(0., 1.)
        if prob <= self.eps:
            ac_real = ac = torch.tensor(
                self.action_space.sample(), dtype=torch.float, device=obs.device)
            q, _ = self.qfunc(obs, ac)
        else:
            q, ac = self.qfunc.max(obs)
            ac_real = self.convert_ac_for_real(ac.detach().cpu().numpy())
        return ac_real, ac, dict(q=q)

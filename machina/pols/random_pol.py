import numpy as np
import torch

from machina.pols import BasePol


class RandomPol(BasePol):
    """
    Policy with uniform distribution.

    Parameters
    ----------
    ob_space : gym.Space
        observation's space
    ac_space : gym.Space
        action's space.
        This should be gym.spaces.Box
    net : torch.nn.Module
    rnn : bool
    normalize_ac : bool
        If True, the output of network is spreaded for ac_space.
        In this situation the output of network is expected to be in -1~1.
    data_parallel : bool
        If True, network computation is executed in parallel.
    parallel_dim : int
        Splitted dimension in data parallel.
    """

    def __init__(self, ob_space, ac_space, net=None, rnn=False, normalize_ac=True, data_parallel=False, parallel_dim=0):
        BasePol.__init__(self, ob_space, ac_space, net, rnn=rnn, normalize_ac=normalize_ac,
                         data_parallel=data_parallel, parallel_dim=parallel_dim)

    def forward(self, ob):
        ac_real = np.random.uniform(
            self.ac_space.low, self.ac_space.high, self.ac_space.shape).astype(np.float32)
        ac = torch.tensor(ac_real)
        mean = torch.zeros_like(ac)
        return ac_real, ac, dict(mean=mean)

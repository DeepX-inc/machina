import torch.nn as nn


class BaseSVfunc(nn.Module):
    """
    Base function of State Value Function.
    It takes observations and then output value.
    For example V Func.

    Parameters
    ----------
    ob_space : gym.Space
    net : torch.nn.Module
    rnn : bool
    data_parallel : bool
        If True, network computation is executed in parallel.
    parallel_dim : int
        Splitted dimension in data parallel.
    """

    def __init__(self, ob_space, net, rnn=False, data_parallel=False, parallel_dim=0):
        nn.Module.__init__(self)
        self.ob_space = ob_space
        self.net = net

        self.rnn = rnn
        self.hs = None

        self.data_parallel = data_parallel
        if data_parallel:
            self.dp_net = nn.DataParallel(self.net, dim=parallel_dim)
        self.dp_run = False

    def reset(self):
        """
        reset for rnn's hidden state.
        """
        if self.rnn:
            self.hs = None

    def _check_obs_shape(self, obs):
        """
        Reshape input appropriately.
        """
        if self.rnn:
            additional_shape = 2
        else:
            additional_shape = 1
        if len(obs.shape) < additional_shape + len(self.ob_space.shape):
            for _ in range(additional_shape + len(self.ob_space.shape) - len(obs.shape)):
                obs = obs.unsqueeze(0)
        return obs

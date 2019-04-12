import torch.nn as nn

from machina.utils import get_device


class BaseSAVfunc(nn.Module):
    """
    Base function of State Action Value Function.
    It takes observations and actions and then output value.
    For example Q Func.

    Parameters
    ----------
    observation_space : gym.Space
    action_space : gym.Space
    net : torch.nn.Module
    rnn : bool
    data_parallel : bool or str
        If True, network computation is executed in parallel.
        If data_parallel is ddp, network computation is executed in distributed parallel.
    parallel_dim : int
        Splitted dimension in data parallel.
    """

    def __init__(self, observation_space, action_space, net, rnn=False, data_parallel=False, parallel_dim=0):
        nn.Module.__init__(self)
        self.observation_space = observation_space
        self.action_space = action_space
        self.net = net

        self.rnn = rnn
        self.hs = None

        self.data_parallel = data_parallel
        if data_parallel:
            if data_parallel is True:
                self.dp_net = nn.DataParallel(self.net, dim=parallel_dim)
            elif data_parallel == 'ddp':
                self.net.to(get_device())
                self.dp_net = nn.parallel.DistributedDataParallel(
                    self.net, device_ids=[get_device()], dim=parallel_dim)
            else:
                raise ValueError(
                    'Bool and str(ddp) are allowed to be data_parallel.')
        self.dp_run = False

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'dp_net' in state['_modules']:
            _modules = copy.deepcopy(state['_modules'])
            del _modules['dp_net']
            state['_modules'] = _modules
        return state

    def __setstate__(self, state):
        if 'dp_net' in state:
            state.pop('dp_net')
        self.__dict__.update(state)

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
        if len(obs.shape) < additional_shape + len(self.observation_space.shape):
            for _ in range(additional_shape + len(self.observation_space.shape) - len(obs.shape)):
                obs = obs.unsqueeze(0)
        return obs

    def _check_acs_shape(self, acs):
        """
        Reshape input appropriately.
        """
        if self.rnn:
            additional_shape = 2
        else:
            additional_shape = 1
        if len(acs.shape) < additional_shape + len(self.action_space.shape):
            for _ in range(additional_shape + len(self.action_space.shape) - len(acs.shape)):
                acs = acs.unsqueeze(0)
        return acs

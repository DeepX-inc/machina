import numpy as np
import torch

from machina.pds import DeterministicPd
from machina.pols import BasePol
from machina.utils import get_device


class MPCPol(BasePol):
    """
    Policy with model predictive control.

    Parameters
    ----------
    ob_space : gym.Space
        observation's space
    ac_space : gym.Space
        action's space.
        This should be gym.spaces.Box
    net : torch.nn.Module
        dymamics model
    rew_func : function
        rt = rew_func(st+1, at). rt, st+1 and at are torch.Tensor.
    env : gym.Env
    n_samples : int
    horizon : int
    rnn : bool
    normalize_ac : bool
        If True, the output of network is spreaded for ac_space.
        In this situation the output of network is expected to be in -1~1.
    data_parallel : bool
        If True, network computation is executed in parallel.
    parallel_dim : int
        Splitted dimension in data parallel.
    """

    def __init__(self, ob_space, ac_space, net, rew_func, env, n_samples=1000, horizon=20,
                 mean_obs=0., std_obs=1., mean_acs=0., std_acs=1., mean_next_obs=0., std_next_obs=1.,
                 rnn=False, normalize_ac=True, data_parallel=False, parallel_dim=0):
        if rnn:
            raise ValueError(
                'rnn with MPCPol is not supported now')
        BasePol.__init__(self, ob_space, ac_space, net, rnn=rnn, normalize_ac=normalize_ac,
                         data_parallel=data_parallel, parallel_dim=parallel_dim)
        if isinstance(env, str):
            env = gym.envs.make(env)
        self.env = env
        self.rew_func = rew_func
        self.n_samples = n_samples
        self.horizon = horizon
        self.to(get_device())

        self.mean_obs = mean_obs.repeat(n_samples, 1).to('cpu')
        self.std_obs = std_obs.repeat(n_samples, 1).to('cpu')
        self.mean_acs = mean_acs.repeat(n_samples, 1).to('cpu')
        self.std_acs = std_acs.to('cpu')
        self.mean_next_obs = mean_next_obs.to('cpu')
        self.std_next_obs = std_next_obs.to('cpu')

    def reset(self):
        super(MPCPol, self).reset()

    def forward(self, ob):
        # randomly sample N candidate action sequences
        sample_acs = np.random.uniform(
            self.ac_space.low[0], self.ac_space.high[0], (self.horizon, self.n_samples, self.ac_space.shape[0]))
        sample_acs = torch.tensor(
            sample_acs, dtype=torch.float)

        # forward simulate the action sequences to get predicted trajectories
        obs = torch.zeros((self.horizon+1, self.n_samples,
                           self.ob_space.shape[0]), dtype=torch.float)
        rews_sum = torch.zeros(
            (self.n_samples), dtype=torch.float)

        obs[0] = ob.repeat(self.n_samples, 1)
        obs[0] = self._check_obs_shape(obs[0])
        with torch.no_grad():
            for i in range(self.horizon):
                ob = (obs[i] - self.mean_obs) / self.std_obs
                ac = (sample_acs[i] - self.mean_acs) / self.std_acs
                # inf to mean
                ob[ob == float('inf')] = 0.
                ac[ac == float('inf')] = 0.
                next_ob = ob + self.net(ob, ac)
                obs[i+1] = next_ob * self.std_next_obs + self.mean_next_obs
                rews_sum += self.rew_func(obs[i+1], sample_acs[i])

        best_sample_index = rews_sum.max(0)[1]
        ac = sample_acs[0][best_sample_index]
        ac_real = ac.cpu().numpy()

        return ac_real, ac, dict(mean=ac)

    def deterministic_ac_real(self, obs):
        """
        action for deployment
        """
        mean_read, mean, dic = self.forward(obs)
        return mean_real, mean, dict(mean=mean)

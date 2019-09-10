import numpy as np
import torch
import copy

from machina.pds import DeterministicPd
from machina.pols import BasePol
from machina.utils import get_device


class MPCPol(BasePol):
    """
    Policy with model predictive control.

    Parameters
    ----------
    observation_space : gym.Space
        observation's space
    action_space : gym.Space
        action's space.
        This should be gym.spaces.Box
    net : torch.nn.Module
        dymamics model
    rew_func : function
        rt = rew_func(st+1, at). rt, st+1 and at are torch.tensor.
    n_samples : int
        num of action samples in the model predictive control
    horizon : int
        horizon of prediction
    mean_obs : np.array
    std_obs : np.array
    mean_acs : np.array
    std_acs : np.array
    rnn : bool
    normalize_ac : bool
        If True, the output of network is spreaded for action_space.
        In this situation the output of network is expected to be in -1~1.
    """

    def __init__(self, observation_space, action_space, net, rew_func, n_samples=1000, horizon=20,
                 mean_obs=0., std_obs=1., mean_acs=0., std_acs=1., rnn=False, normalize_ac=True):
        BasePol.__init__(self, observation_space, action_space,
                         net, rnn=rnn, normalize_ac=normalize_ac)
        self.rew_func = rew_func
        self.n_samples = n_samples
        self.horizon = horizon
        self.to(get_device())

        self.mean_obs = torch.tensor(
            mean_obs, dtype=torch.float).repeat(n_samples, 1)
        self.std_obs = torch.tensor(
            std_obs, dtype=torch.float).repeat(n_samples, 1)
        self.mean_acs = torch.tensor(
            mean_acs, dtype=torch.float).repeat(n_samples, 1)
        self.std_acs = torch.tensor(
            std_acs, dtype=torch.float).repeat(n_samples, 1)

    def reset(self):
        super(MPCPol, self).reset()

    def forward(self, ob, hs=None, h_masks=None):
        # randomly sample N candidate action sequences
        sample_acs = torch.empty(self.horizon, self.n_samples, self.action_space.shape[0], dtype=torch.float).uniform_(
            self.action_space.low[0], self.action_space.high[0])
        normalized_acs = (sample_acs - self.mean_acs) / self.std_acs

        # forward simulate the action sequences to get predicted trajectories
        obs = torch.zeros((self.horizon + 1, self.n_samples,
                           self.observation_space.shape[0]), dtype=torch.float)
        rews_sum = torch.zeros(
            (self.n_samples), dtype=torch.float)
        obs[0] = ob.repeat(self.n_samples, 1)
        obs[0] = (obs[0] - self.mean_obs) / self.std_obs

        if self.rnn:
            time_seq, batch_size, *_ = obs.shape

            if hs is None:
                if self.hs is None:
                    self.hs = self.net.init_hs(batch_size)
                hs = self.hs

            if h_masks is None:
                h_masks = hs[0].new(time_seq, batch_size, 1).zero_()
            h_masks = h_masks.reshape(time_seq, batch_size, 1)

        with torch.no_grad():
            for i in range(self.horizon):
                ac = normalized_acs[i]
                if self.rnn:
                    d_ob, hs = self.net(obs[i].unsqueeze(
                        0), ac.unsqueeze(0), hs, h_masks)
                    obs[i + 1] = obs[i] + d_ob
                else:
                    obs[i + 1] = obs[i] + self.net(obs[i], ac)
                rews_sum += self.rew_func(obs[i + 1], sample_acs[i],
                                          self.mean_obs, self.std_obs)

        best_sample_index = rews_sum.max(0)[1]
        ac = sample_acs[0][best_sample_index]
        ac_real = ac.cpu().numpy()

        if self.rnn:
            normalized_ac = normalized_acs[0][best_sample_index].repeat(
                self.n_samples, 1)
            with torch.no_grad():
                _, self.hs = self.net(obs[0].unsqueeze(
                    0), normalized_ac.unsqueeze(0), self.hs, h_masks)

        return ac_real, ac, dict(mean=ac)

    def deterministic_ac_real(self, obs):
        """
        action for deployment
        """
        mean_real, mean, dic = self.forward(obs)
        return mean_real, mean, dic

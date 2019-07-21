"""
Deterministic State Action Value function with Cross Entropy Method
"""

from machina.vfuncs.state_action_vfuncs.deterministic_state_action_vfunc import DeterministicSAVfunc
from machina.utils import get_device
import torch
from torch.distributions import Normal, MultivariateNormal


class CEMDeterministicSAVfunc(DeterministicSAVfunc):
    """
    Deterministic State Action Vfunction with Cross Entropy Method.

    Parameters
    ----------
    observation_space : gym.Space
    action_space : gym.Space
    net : torch.nn.Module
    rnn : bool
    num_sampling : int
        Number of samples sampled from Gaussian in CEM.
    num_best_sampling : int
        Number of best samples used for fitting Gaussian in CEM.
    num_iter : int
        Number of iteration of CEM.
    delta : float
        Coefficient used for making covariance matrix positive definite.
    """

    def __init__(self, observation_space, action_space, net, rnn=False, num_sampling=64,
                 num_best_sampling=6, num_iter=2, multivari=True, delta=1e-4, save_memory=False):
        super().__init__(observation_space, action_space, net, rnn)
        self.num_sampling = num_sampling
        self.delta = delta
        self.num_best_sampling = num_best_sampling
        self.num_iter = num_iter
        self.net = net
        self.dim_ac = self.action_space.shape[0]
        self.multivari = multivari
        self.save_memory = save_memory
        self.to(get_device())

    def max(self, obs):
        """
        Perform max and argmax of Qfunc

        Parameters
        ----------
        obs : torch.Tensor

        Returns
        -------
        max_qs : torch.Tensor
        max_acs : torch.Tensor
        """

        obs = self._check_obs_shape(obs)

        self.dim_ob = obs.shape[1]
        high = torch.tensor(self.action_space.high,
                            dtype=torch.float, device=get_device())
        low = torch.tensor(
            self.action_space.low, dtype=torch.float, device=get_device())
        init_samples = torch.linspace(
            0, 1, self.num_sampling, device=get_device())
        init_samples = init_samples.reshape(
            self.num_sampling, -1) * (high - low) + low  # (self.num_sampling, dim_ac)
        init_samples = self._clamp(init_samples)
        if not self.save_memory:  # batch
            self.cem_batch_size = obs.shape[0]
            obs = obs.repeat((1, self.num_sampling)).reshape(
                (self.cem_batch_size * self.num_sampling, self.dim_ob))
            # concatenate[(self.num_sampling, dim_ac), ..., (self.num_sampling, self.dim_ob)], dim=0)
            init_samples = init_samples.repeat((self.cem_batch_size, 1))
            # concatenate[(self.num_sampling, dim_ac), ..., (self.num_sampling, dim_ac)], dim=0)
            max_qs, max_acs = self._cem(obs, init_samples)
        else:  # for-sentence
            self.cem_batch_size = 1
            max_acs = []
            max_qs = []
            for ob in obs:
                ob = ob.repeat((1, self.num_sampling)).reshape(
                    (self.cem_batch_size * self.num_sampling, self.dim_ob))
                ob = self._check_obs_shape(ob)
                max_q, max_ac = self._cem(ob, init_samples)
                max_qs.append(max_q)
                max_acs.append(max_ac)
            max_qs = torch.tensor(
                max_qs, dtype=torch.float, device=obs.device)
            max_acs = torch.cat(max_acs, dim=0)
        max_acs = self._check_acs_shape(max_acs)
        return max_qs, max_acs

    def _cem(self, obs, samples):
        """
        Perform cross entropy method

        Parameters
        ----------
        obs : torch.Tensor
        samples : torch.Tensor
            shape (self.num_sampling, dim_ac)

        Returns
        -------
        max_q : torch.Tensor
        max_ac : torch.Tensor
        """
        for i in range(self.num_iter + 1):
            with torch.no_grad():
                qvals, _ = self.forward(obs, samples)
            if i != self.num_iter:
                qvals = qvals.reshape((self.cem_batch_size, self.num_sampling))
                _, indices = torch.sort(qvals, dim=1, descending=True)
                best_indices = indices[:, :self.num_best_sampling]
                best_indices = best_indices + \
                    torch.arange(0, self.num_sampling * self.cem_batch_size,
                                 self.num_sampling, device=get_device()).reshape((self.cem_batch_size, 1))
                best_indices = best_indices.reshape(
                    (self.num_best_sampling * self.cem_batch_size,))
                # (self.num_best_sampling * self.cem_batch_size,  self.dim_ac)
                best_samples = samples[best_indices, :]
                # (self.cem_batch_size, self.num_best_sampling, self.dim_ac)
                best_samples = best_samples.reshape(
                    (self.cem_batch_size, self.num_best_sampling, self.dim_ac))
                samples = self._fitting_diag(
                    best_samples) if not self.multivari else self._fitting_multivari(best_samples)
        qvals = qvals.reshape((self.cem_batch_size, self.num_sampling))
        samples = samples.reshape(
            (self.cem_batch_size, self.num_sampling, self.dim_ac))
        max_q, ind = torch.max(qvals, dim=1)
        max_ac = samples[torch.arange(
            self.cem_batch_size, device=get_device()), ind]
        return max_q, max_ac

    def _fitting_diag(self, best_samples):
        """
        Fit diagonal covariance gaussian and sampling from it

        Parameters
        ----------
        best_samples : torch.Tensor
            shape (self.cem_batch_size, self.num_best_sampling, self.dim_ac)

        Returns
        -------
        samples : torch.Tensor
        """
        mean = torch.mean(
            best_samples, dim=1)  # (self.cem_batch_size, self.dim_ac)
        # (self.cem_batch_size, self.dim_ac)
        std = torch.std(best_samples, dim=1)
        samples = Normal(loc=mean, scale=std).rsample(
            torch.Size((self.num_sampling,)))  # (self.num_best_sampling, self.cem_batch_size, self.dim_ac)
        # (self.num_best_sampling, self.cem_batch_size, self.dim_ac)
        samples = samples.transpose(1, 0)
        samples = samples.reshape((self.num_sampling * self.cem_batch_size,
                                   self.dim_ac))  # (self.num_best_sampling * self.cem_batch_size,  self.dim_ac)
        # (self.num_best_sampling * self.cem_batch_size,  self.dim_ac)
        samples = self._clamp(samples)
        return samples

    def _fitting_multivari(self, best_samples):
        """
        Fit multivariate gaussian and sampling from it

        Parameters
        ----------
        best_samples : torch.Tensor
            shape (self.cem_batch_size, self.num_best_sampling, self.dim_ac)

        Returns
        -------
        samples : torch.Tensor
        """
        def fitting(best_samples):
            mean = best_samples.mean(dim=0)
            fs_m = best_samples.sub(mean.expand_as(best_samples))
            cov_mat = fs_m.transpose(0, 1).mm(fs_m) / (self.num_sampling - 1)
            cov_mat = cov_mat + self.delta * torch.eye(cov_mat.shape[0])
            pd = MultivariateNormal(mean, cov_mat)
            samples = pd.sample((self.num_sampling,))
            return samples
        samples = torch.cat([fitting(best_sample)
                             for best_sample in best_samples], dim=0)
        return samples

    def _clamp(self, samples):
        low = torch.tensor(self.action_space.low,
                           dtype=torch.float, device=get_device())
        high = torch.tensor(self.action_space.high,
                            dtype=torch.float, device=get_device())
        samples = (samples - low) / (high - low)
        samples = torch.clamp(samples, 0, 1) * (high - low) + low
        return samples

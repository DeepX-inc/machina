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
    ob_space : gym.Space
    ac_space : gym.Space
    net : torch.nn.Module
    rnn : bool
    data_parallel : bool
        If True, network computation is executed in parallel.
    parallel_dim : int
        Splitted dimension in data parallel.
    num_sampling : int
        Number of samples sampled from Gaussian in CEM.
    num_best_sampling : int
        Number of best samples used for fitting Gaussian in CEM.
    num_iter : int
        Number of iteration of CEM.
    delta : float
        Coefficient used for making covariance matrix positive definite.
    """

    def __init__(self, ob_space, ac_space, net, rnn=False, data_parallel=False, parallel_dim=0, num_sampling=64, num_best_sampling=6, num_iter=2, multivari=True, delta=1e-4):
        super().__init__(ob_space, ac_space, net, rnn, data_parallel, parallel_dim)
        self.num_sampling = num_sampling
        self.delta = delta
        self.num_best_sampling = num_best_sampling
        self.num_iter = num_iter
        self.net = net
        self.dim_ac = self.ac_space.shape[0]
        self.multivari = multivari
        self.to(get_device())

    def max(self, obs):
        """
        Max and Argmax of Qfunc
        Parameters
        ----------
        obs : torch.Tensor

        Returns
        -------
        max_qs, max_acs
        """

        obs = self._check_obs_shape(obs)

        self.batch_size = obs.shape[0]
        self.dim_ob = obs.shape[1]
        high = torch.tensor(self.ac_space.high,
                            dtype=torch.float, device=get_device())
        low = torch.tensor(
            self.ac_space.low, dtype=torch.float, device=get_device())
        init_samples = torch.linspace(0, 1, self.num_sampling, device=get_device()).reshape(
            self.num_sampling, -1) * (high - low) + low  # (self.num_sampling, dim_ac)
        init_samples = self._clamp(init_samples)
        max_qs, max_acs = self._cem(obs, init_samples)
        return max_qs, max_acs

    def _cem(self, obs, init_samples):
        """

        Parameters
        ----------
        obs : torch.Tensor
        init_samples : torch.Tensor
            shape (self.num_sampling, dim_ac)
        Returns
        -------

        """
        obs = obs.repeat((1, self.num_sampling)).reshape(
            (self.num_sampling * self.batch_size, self.dim_ob))
        samples = init_samples.repeat((self.batch_size, 1))
        for i in range(self.num_iter):
            with torch.no_grad():
                qvals, _ = self.forward(obs, samples)
            if i != self.num_iter-1:
                qvals = qvals.reshape((self.batch_size, self.num_sampling))
                _, indices = torch.sort(qvals, dim=1, descending=True)
                best_indices = indices[:, :self.num_best_sampling]
                best_indices = best_indices + \
                    torch.arange(0, self.num_sampling*self.batch_size,
                                 self.num_sampling, device=get_device()).reshape(self.batch_size, 1)
                best_indices = best_indices.reshape(
                    (self.num_best_sampling * self.batch_size,))
                # (self.num_best_sampling * self.batch_size,  self.dim_ac)
                best_samples = samples[best_indices, :]
                # (self.batch_size, self.num_best_sampling, self.dim_ac)
                best_samples = best_samples.reshape(
                    (self.batch_size, self.num_best_sampling, self.dim_ac))
                samples = self._fitting_diag(
                    best_samples) if not self.multivari else self._fitting_multivari(best_samples)
        qvals = qvals.reshape((self.batch_size, self.num_sampling))
        samples = samples.reshape(
            (self.batch_size, self.num_sampling, self.dim_ac))
        max_q, ind = torch.max(qvals, dim=1)
        max_ac = samples[torch.arange(
            self.batch_size, device=get_device()), ind]
        max_ac = self._check_acs_shape(max_ac)
        return max_q, max_ac

    def _fitting_diag(self, best_samples):
        """
        fitting diagonal covariance gaussian and sampling from it
        Parameters
        ----------
        best_samples : torch.Tensor
            shape (self.batch_size, self.num_best_sampling, self.dim_ac)

        Returns
        -------
        samples : torch.Tensor
        """
        mean = torch.mean(
            best_samples, dim=1)  # (self.batch_size, self.dim_ac)
        std = torch.std(best_samples, dim=1)  # (self.batch_size, self.dim_ac)
        samples = Normal(loc=mean, scale=std).rsample(
            torch.Size((self.num_sampling,)))  # (self.num_best_sampling, self.batch_size, , self.dim_ac)
        # (self.num_best_sampling, self.batch_size, self.dim_ac)
        samples = samples.transpose(1, 0)
        samples = samples.reshape((self.num_sampling * self.batch_size,
                                   self.dim_ac))  # (self.num_best_sampling * self.batch_size,  self.dim_ac)
        # (self.num_best_sampling * self.batch_size,  self.dim_ac)
        samples = self._clamp(samples)
        return samples

    def _fitting_multivari(self, best_samples):
        """
        fitting multivariate gaussian and sampling from it
        Parameters
        ----------
        best_samples : torch.Tensor
            shape (self.batch_size, self.num_best_sampling, self.dim_ac)

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
        low = torch.tensor(self.ac_space.low,
                           dtype=torch.float, device=get_device())
        high = torch.tensor(self.ac_space.high,
                            dtype=torch.float, device=get_device())
        samples = (samples - low) / (high - low)
        samples = torch.clamp(samples, 0, 1) * (high - low) + low
        return samples

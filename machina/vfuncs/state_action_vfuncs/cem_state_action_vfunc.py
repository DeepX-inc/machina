"""
Deterministic State Action Value function with Cross Entropy Method
"""

from machina.vfuncs.state_action_vfuncs.deterministic_state_action_vfunc import DeterministicSAVfunc
from machina.utils import get_device
import torch
from torch.distributions import MultivariateNormal


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

    def __init__(self, ob_space, ac_space, net, rnn=False, data_parallel=False, parallel_dim=0, num_sampling=64, num_best_sampling=6, num_iter=2, delta=1e-4):
        super().__init__(ob_space, ac_space, net, rnn, data_parallel, parallel_dim)
        self.num_sampling = num_sampling
        self.delta = delta
        self.num_best_sampling = num_best_sampling
        self.num_iter = num_iter
        self.net = net
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
        high = torch.tensor(self.ac_space.high,
                            dtype=torch.float, device=obs.device)
        low = torch.tensor(
            self.ac_space.low, dtype=torch.float, device=obs.device)
        pd = MultivariateNormal((high - low)/2.,
                                torch.eye(self.ac_space.shape[0]))
        init_samples = pd.sample((self.num_sampling,))
        init_samples = self._clamp(init_samples)

        if obs.dim() == 1:
            # when sampling policy
            obs = obs.unsqueeze(0)
        max_qs, max_acs = self._cem(obs, init_samples)
        return max_qs, max_acs

    def _cem(self, obs, samples):
        batch_size = obs.shape[0]
        dim_ob = obs.shape[1]
        obs = obs.repeat((1, self.num_sampling)).reshape(
            (self.num_sampling * batch_size, dim_ob))
        for i in range(self.num_iter):
            with torch.no_grad():
                # shape (self.num_sampling * batch_size, dim_ac)
                samples = samples.repeat((batch_size, 1))
                qvals, _ = self.forward(obs, samples)
            if i != self.num_iter-1:
                qvals = qvals.reshape((batch_size, self.num_sampling))
                _, indices = torch.sort(qvals, dim=1, descending=True)
                best_indices = indices[:, :self.num_best_sampling]
                best_indices = best_indices + \
                    torch.arange(0, self.num_sampling*batch_size,
                                 self.num_sampling).view(batch_size, 1)
                best_indices = best_indices.view(
                    (self.num_best_sampling * batch_size,))
                best_samples = samples[best_indices, :]
                samples = self._fitting(best_samples)
                samples = self._clamp(samples)
        qvals = qvals.reshape((batch_size, self.num_sampling))
        samples = samples.reshape(
            (batch_size, self.num_sampling, self.ac_space.shape[0]))
        max_q, ind = torch.max(qvals, dim=1)
        max_ac = samples[torch.arange(batch_size), ind]
        return max_q, max_ac.view((batch_size, -1))

    def _fitting(self, fitting_samples):
        """
        fitting gaussian and sampling from it
        Parameters
        ----------
        fitting_samples : torch.Tensor
            shape (self.num_best_sampling, dim_ac)

        Returns
        -------
        samples : torch.Tensor
        """
        mean = fitting_samples.mean(dim=0)
        fs_m = fitting_samples.sub(mean.expand_as(fitting_samples))
        cov_mat = fs_m.transpose(0, 1).mm(fs_m) / (self.num_sampling - 1)
        cov_mat = cov_mat + self.delta * torch.eye(cov_mat.shape[0])
        pd = MultivariateNormal(mean, cov_mat)
        samples = pd.sample((self.num_sampling,))
        return samples

    def _clamp(self, samples):
        low = torch.tensor(self.ac_space.low,
                           dtype=torch.float, device=samples.device)
        high = torch.tensor(self.ac_space.high,
                            dtype=torch.float, device=samples.device)
        samples = (samples - low) / (high - low)
        samples = torch.clamp(samples, 0, 1) * (high - low) + low
        return samples

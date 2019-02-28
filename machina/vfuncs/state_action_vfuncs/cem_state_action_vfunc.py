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
        :param ob:
        :return:
        """
        max_ac = torch.tensor(self.ac_space.high,
                              dtype=torch.float, device=get_device())
        min_ac = torch.tensor(
            self.ac_space.low, dtype=torch.float, device=get_device())
        pd = MultivariateNormal((max_ac - min_ac)/2.,
                                torch.eye(self.ac_space.shape[0]))
        init_samples = pd.sample((self.num_sampling,))
        init_samples = self._clamp(init_samples)

        if obs.dim() == 1:
            # when sampling policy
            max_qs, max_acs = self._cem(obs, init_samples)
        else:
            # when training policy
            max_acs = []
            max_qs = []
            for ob in obs:
                max_q, max_ac = self._cem(ob, init_samples)
                max_qs.append(max_q)
                max_acs.append(max_ac.unsqueeze(0))
            max_qs = torch.tensor(
                max_qs, dtype=torch.float, device=get_device())
            max_acs = torch.cat(max_acs, dim=0)
        return max_qs, max_acs

    def _cem(self, ob, samples):
        for i in range(self.num_iter):
            with torch.no_grad():
                qvals, _ = self.forward(
                    ob.expand((self.num_sampling,  -1)), samples)
            if i != self.num_iter-1:
                _, indices = torch.sort(qvals, descending=True)
                best_indices = indices[:self.num_best_sampling]
                best_samples = samples[best_indices, :]
                samples = self._fitting(best_samples)
                samples = self._clamp(samples)
        max_q = torch.max(qvals)
        ind = torch.argmax(qvals)
        max_ac = samples[ind]
        return max_q, max_ac

    def _fitting(self, fitting_samples):
        """
        :param fitting_samples:
        :return: fitted multivariate gaussian
        """
        mean = fitting_samples.mean(dim=0)
        fs_m = fitting_samples.sub(mean.expand_as(fitting_samples))
        cov_mat = fs_m.transpose(0, 1).mm(fs_m) / (self.num_sampling - 1)
        cov_mat = cov_mat + self.delta * torch.eye(cov_mat.shape[0])
        pd = MultivariateNormal(mean, cov_mat)
        samples = pd.sample((self.num_sampling,))
        return samples

    def _clamp(self, samples):
        for i, (ac_high, ac_low) in enumerate(zip(self.ac_space.high, self.ac_space.low)):
            samples[i] = torch.clamp(samples[i], ac_high, ac_low)
        return samples

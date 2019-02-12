"""
Deterministic State Action Valu function
"""

from machina.pds import DeterministicPd
from machina.vfuncs.state_action_vfuncs.base import BaseSAVfunc
from machina.utils import get_device


class DeterministicSAVfunc(BaseSAVfunc):
    """
    Deterministic version of State Action Value Function.

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
    """

    def __init__(self, ob_space, ac_space, net, rnn=False, data_parallel=False, parallel_dim=0):
        super().__init__(ob_space, ac_space, net, rnn, data_parallel, parallel_dim)
        self.pd = DeterministicPd()
        self.to(get_device())

    def forward(self, obs, acs, hs=None, h_masks=None):
        """
        Calculating values.
        """
        obs = self._check_obs_shape(obs)
        acs = self._check_acs_shape(acs)

        if self.rnn:
            time_seq, batch_size, *_ = obs.shape

            if hs is None:
                if self.hs is None:
                    self.hs = self.net.init_hs(batch_size)
                if self.dp_run:
                    self.hs = (self.hs[0].unsqueeze(
                        0), self.hs[1].unsqueeze(0))
                hs = self.hs

            if h_masks is None:
                h_masks = hs[0].new(time_seq, batch_size, 1).zero_()
            h_masks = h_masks.reshape(time_seq, batch_size, 1)

            if self.dp_run:
                vs, hs = self.dp_net(obs, acs, hs, h_masks)
            else:
                vs, hs = self.net(obs, acs, hs, h_masks)
            self.hs = hs
        else:
            if self.dp_run:
                vs = self.dp_net(obs, acs)
            else:
                vs = self.net(obs, acs)
        return vs.squeeze(-1), dict(mean=vs.squeeze(-1), hs=hs)

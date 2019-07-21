"""
Deterministic State Dynamics Model
"""

from machina.models.base import BaseModel
from machina.utils import get_device


class DeterministicSModel(BaseModel):
    """
    Deterministic version of State Dynamics Model.

    Parameters
    ----------
    observation_space : gym.Space
    action_space : gym.Space
    net : torch.nn.Module
    rnn : bool
    """

    def __init__(self, observation_space, action_space, net, rnn=False):
        super().__init__(observation_space, action_space, net, rnn)
        self.to(get_device())

    def forward(self, obs, acs, hs=None, h_masks=None):
        obs = self._check_obs_shape(obs)
        acs = self._check_acs_shape(acs)

        if self.rnn:
            time_seq, batch_size, *_ = obs.shape

            if hs is None:
                if self.hs is None:
                    self.hs = self.net.init_hs(batch_size)
                hs = self.hs

            if h_masks is None:
                h_masks = hs[0].new(time_seq, batch_size, 1).zero_()
            h_masks = h_masks.reshape(time_seq, batch_size, 1)

            d_ob, hs = self.net(obs, acs, hs, h_masks)
            self.hs = hs
        else:
            d_ob = self.net(obs, acs)
        return d_ob, dict(mean=d_ob)

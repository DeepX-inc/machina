"""
Recurrent State-Space Model
"""

import torch
from torch import nn

from machina.models.base import BaseModel
from machina.pds.gaussian_pd import GaussianPd
from machina.utils import get_device


class RecurrentSSpaceModel(BaseModel):
    """
    Recurrent State Space Model.

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

    def __init__(self, ob_space, ac_space, embed_size, state_size, belief_size, hidden_size, min_stddev=1e-1, data_parallel=False, parallel_dim=0):
        super().__init__(ob_space, ac_space, None, True, data_parallel, parallel_dim)
        self.embed_size = embed_size
        self.state_size = state_size
        self.belief_size = belief_size
        self.hidden_size = hidden_size
        self.hs = None
        self.device = 'cpu'

        self.encoder1 = nn.Linear(ob_space.shape[0], hidden_size)
        self.encoder2 = nn.Linear(hidden_size, hidden_size)
        self.encoder3 = nn.Linear(hidden_size, embed_size)

        self.fc1 = nn.Linear(state_size + ac_space.shape[0], hidden_size)
        self.cell = nn.GRUCell(hidden_size, belief_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, state_size)
        self.fc4 = nn.Linear(hidden_size, state_size)
        self.softplus1 = nn.Softplus()
        self.eye = torch.eye(state_size)
        self.pd = GaussianPd()

        self.fc5 = nn.Linear(hidden_size+embed_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, state_size)
        self.fc7 = nn.Linear(hidden_size, state_size)
        self.softplus2 = nn.Softplus()
        self.to(get_device())

    def to(self, device):
        super().to(device)
        self.device = device

    def init_hs(self, batch_size=1):
        new_hs = torch.zeros(batch_size, self.belief_size,
                             dtype=torch.float, device=self.device)
        return new_hs

    def encode(self, obs):
        hidden = torch.relu(self.encoder1(obs))
        hidden = torch.relu(self.encoder2(hidden))
        embedded_obs = self.encoder3(hidden)
        return embedded_obs

    def prior(self, prev_state, prev_action, hs=None):
        """Compute prior next state by applying the transition dynamics."""
        batch_size, *_ = prev_state.shape

        if hs is None:
            if self.hs is None:
                self.hs = self.init_hs(batch_size)
            hs = self.hs

        inputs = torch.cat([prev_state, prev_action], -1)
        hidden = torch.relu(self.fc1(inputs))
        belief = torch.relu(self.cell(hidden, hs))
        self.hs = belief

        hidden = torch.relu(self.fc2(belief))
        mean = self.fc3(hidden)
        log_std = torch.log(self.softplus1(self.fc4(hidden)))

        sample = self.pd.sample(dict(mean=mean, log_std=log_std))

        return {
            'mean': mean,
            'log_std': log_std,
            'sample': sample,
            'belief': belief,
        }

    def posterior(self, prev_state, prev_action, obs, hs=None):
        """Compute posterior state from previous state and current observation."""
        batch_size, *_ = prev_state.shape

        prior = self.prior(prev_state, prev_action, hs)

        inputs = torch.cat([prior['belief'], obs], -1)
        hidden = torch.relu(self.fc5(inputs))
        mean = self.fc6(hidden)
        log_std = torch.log(self.softplus2(self.fc7(hidden)))

        sample = self.pd.sample(dict(mean=mean, log_std=log_std))

        return {
            'mean': mean,
            'log_std': log_std,
            'sample': sample,
            'belief': prior['belief'],
        }

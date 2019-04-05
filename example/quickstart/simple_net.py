"""
Examples of network architecture.
"""

import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, uniform_
import torch.nn.functional as F
import gym


def mini_weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(uniform_(m.weight.data, -3e-3, 3e-3))
        m.bias.data.fill_(0)


def weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(kaiming_uniform_(m.weight.data))
        m.bias.data.fill_(0)


class PolNet(nn.Module):
    def __init__(self, observation_space, action_space, h1=200, h2=100, deterministic=False):
        super(PolNet, self).__init__()

        self.deterministic = deterministic

        if isinstance(action_space, gym.spaces.Box):
            self.discrete = False
        else:
            self.discrete = True
            if isinstance(action_space, gym.spaces.MultiDiscrete):
                self.multi = True
            else:
                self.multi = False

        self.fc1 = nn.Linear(observation_space.shape[0], h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)

        if not self.discrete:
            self.mean_layer = nn.Linear(h2, action_space.shape[0])
            if not self.deterministic:
                self.log_std_param = nn.Parameter(
                    torch.randn(action_space.shape[0])*1e-10 - 1)
            self.mean_layer.apply(mini_weight_init)
        else:
            if self.multi:
                self.output_layers = nn.ModuleList(
                    [nn.Linear(h2, vec) for vec in action_space.nvec])
                list(map(lambda x: x.apply(mini_weight_init), self.output_layers))
            else:
                self.output_layer = nn.Linear(h2, action_space.n)
                self.output_layer.apply(mini_weight_init)

    def forward(self, ob):
        h = F.relu(self.fc1(ob))
        h = F.relu(self.fc2(h))
        if not self.discrete:
            mean = torch.tanh(self.mean_layer(h))
            if not self.deterministic:
                log_std = self.log_std_param.expand_as(mean)
                return mean, log_std
            else:
                return mean
        else:
            if self.multi:
                return torch.cat([torch.softmax(ol(h), dim=-1).unsqueeze(-2) for ol in self.output_layers], dim=-2)
            else:
                return torch.softmax(self.output_layer(h), dim=-1)


class VNet(nn.Module):
    def __init__(self, observation_space, h1=200, h2=100):
        super(VNet, self).__init__()
        self.fc1 = nn.Linear(observation_space.shape[0], h1)
        self.fc2 = nn.Linear(h1, h2)
        self.output_layer = nn.Linear(h2, 1)
        self.apply(weight_init)

    def forward(self, ob):
        h = F.relu(self.fc1(ob))
        h = F.relu(self.fc2(h))
        return self.output_layer(h)


class QNet(nn.Module):
    def __init__(self, observation_space, action_space, h1=300, h2=400):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(observation_space.shape[0], h1)
        self.fc2 = nn.Linear(action_space.shape[0] + h1, h2)
        self.output_layer = nn.Linear(h2, 1)
        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)
        self.output_layer.apply(mini_weight_init)

    def forward(self, ob, ac):
        h = F.relu(self.fc1(ob))
        h = torch.cat([h, ac], dim=-1)
        h = F.relu(self.fc2(h))
        return self.output_layer(h)


class PolNetLSTM(nn.Module):
    def __init__(self, observation_space, action_space, h_size=1024, cell_size=512):
        super(PolNetLSTM, self).__init__()
        self.h_size = h_size
        self.cell_size = cell_size
        self.rnn = True

        if isinstance(action_space, gym.spaces.Box):
            self.discrete = False
        else:
            self.discrete = True
            if isinstance(action_space, gym.spaces.MultiDiscrete):
                self.multi = True
            else:
                self.multi = False

        self.input_layer = nn.Linear(observation_space.shape[0], self.h_size)
        self.cell = nn.LSTMCell(self.h_size, hidden_size=self.cell_size)
        if not self.discrete:
            self.mean_layer = nn.Linear(self.cell_size, action_space.shape[0])
            self.log_std_param = nn.Parameter(
                torch.randn(action_space.shape[0])*1e-10 - 1)

            self.mean_layer.apply(mini_weight_init)
        else:
            if self.multi:
                self.output_layers = nn.ModuleList(
                    [nn.Linear(self.cell_size, vec) for vec in action_space.nvec])
                list(map(lambda x: x.apply(mini_weight_init), self.output_layers))
            else:
                self.output_layer = nn.Linear(self.cell_size, action_space.n)
                self.output_layer.apply(mini_weight_init)

    def init_hs(self, batch_size=1):
        new_hs = (next(self.parameters()).new(batch_size, self.cell_size).zero_(), next(
            self.parameters()).new(batch_size, self.cell_size).zero_())
        return new_hs

    def forward(self, xs, hs, h_masks):
        time_seq, batch_size, *_ = xs.shape

        hs = (hs[0].reshape(batch_size, self.cell_size),
              hs[1].reshape(batch_size, self.cell_size))

        xs = torch.relu(self.input_layer(xs))

        hiddens = []
        for x, mask in zip(xs, h_masks):
            hs = (hs[0] * (1 - mask), hs[1] * (1 - mask))
            hs = self.cell(x, hs)
            hiddens.append(hs[0])
        hiddens = torch.cat([h.unsqueeze(0) for h in hiddens], dim=0)

        if not self.discrete:
            means = torch.tanh(self.mean_layer(hiddens))
            log_std = self.log_std_param.expand_as(means)
            return means, log_std, hs
        else:
            if self.multi:
                return torch.cat([torch.softmax(ol(hiddens), dim=-1).unsqueeze(-2) for ol in self.output_layers], dim=-2), hs
            else:
                return torch.softmax(self.output_layer(hiddens), dim=-1), hs


class VNetLSTM(nn.Module):
    def __init__(self, observation_space, h_size=1024, cell_size=512):
        super(VNetLSTM, self).__init__()
        self.h_size = h_size
        self.cell_size = cell_size
        self.rnn = True

        self.input_layer = nn.Linear(observation_space.shape[0], self.h_size)
        self.cell = nn.LSTMCell(self.h_size, hidden_size=self.cell_size)
        self.output_layer = nn.Linear(self.cell_size, 1)

        self.output_layer.apply(mini_weight_init)

    def init_hs(self, batch_size=1):
        new_hs = (next(self.parameters()).new(batch_size, self.cell_size).zero_(), next(
            self.parameters()).new(batch_size, self.cell_size).zero_())
        return new_hs

    def forward(self, xs, hs, h_masks):
        time_seq, batch_size, *_ = xs.shape

        hs = (hs[0].reshape(batch_size, self.cell_size),
              hs[1].reshape(batch_size, self.cell_size))

        xs = torch.relu(self.input_layer(xs))

        hiddens = []
        for x, mask in zip(xs, h_masks):
            hs = (hs[0] * (1 - mask), hs[1] * (1 - mask))
            hs = self.cell(x, hs)
            hiddens.append(hs[0])
        hiddens = torch.cat([h.unsqueeze(0) for h in hiddens], dim=0)
        outs = self.output_layer(hiddens)

        return outs, hs

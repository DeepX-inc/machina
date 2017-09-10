
# TODO: remove
import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import machina as mc
from machina.pols import GaussianPol
from machina.algos import ppo
from machina.prepro import BasePrePro
from machina.vfuncs import DeterministicVfunc
from machina.data import BaseData
from machina.samplers import BatchSampler
from machina.misc import logger

env = gym.make('Pendulum-v0')
ob_space = env.observation_space
ac_space = env.action_space

class PolNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(ob_space.shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean_layer = nn.Linear(64, 1)
        def mini_weight_init(m):
            if m.__class__.__name__ == 'Linear':
                m.weight.data.normal_()
                norm = m.weight.data.pow(2).sum(dim=0, keepdim=True).sqrt()
                m.weight.data.mul_(0.01/norm)
                m.bias.data.fill_(0)
        def weight_init(m):
            if m.__class__.__name__ == 'Linear':
                m.weight.data.normal_()
                norm = m.weight.data.pow(2).sum(dim=0, keepdim=True).sqrt()
                m.weight.data.mul_(1/norm)
                m.bias.data.fill_(0)
        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)
        self.mean_layer.apply(mini_weight_init)
        self.log_std_param = nn.Parameter(torch.zeros(1))
    def forward(self, ob):
        h = F.tanh(self.fc1(ob))
        h = F.tanh(self.fc2(h))
        mean = self.mean_layer(h)
        return mean, self.log_std_param

class VNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(ob_space.shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)
        def weight_init(m):
            if m.__class__.__name__ == 'Linear':
                m.weight.data.normal_()
                norm = m.weight.data.pow(2).sum(dim=0, keepdim=True).sqrt()
                m.weight.data.mul_(1/norm)
                m.bias.data.fill_(0)
        self.apply(weight_init)

    def forward(self, ob):
        h = F.tanh(self.fc1(ob))
        h = F.tanh(self.fc2(h))
        return self.output_layer(h)

pol_net = PolNet()
pol = GaussianPol(ob_space, ac_space, pol_net)
v_net = VNet()
vfunc = DeterministicVfunc(ob_space, v_net)
prepro = BasePrePro(ob_space)
sampler = BatchSampler(env)
mc_optim = torch.optim.Adam(list(pol_net.parameters())+list(v_net.parameters()), 3e-4)

for epoch in range(10000):
   #paths = sampler.sample(pol, 2048, prepro=prepro.prepro_with_update)
   paths = sampler.sample(pol, 2048)
   logger.record_tabular('Reward', np.mean([np.sum(path['rews']) for path in paths]))
   data = BaseData(paths, shuffle=False)
   ppo.train(data, pol, vfunc, 0.2, mc_optim, 10, 64, 0.99, 0.95)
   logger.dump_tabular()


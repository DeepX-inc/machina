import torch

from machina.vfuncs.base import BaseVfunc
from machina.utils import get_gpu, Variable

class DeterministicVfunc(BaseVfunc):
    def __init__(self, ob_space, net):
        BaseVfunc.__init__(self, ob_space)
        self.net = net
        gpu_id = get_gpu()
        if gpu_id != -1:
            self.cuda(gpu_id)

    def forward(self, obs):
        return self.net(obs).view(-1)


class NormalizedDeterministicVfunc(DeterministicVfunc):
    def __init__(self, ob_space, net):
        DeterministicVfunc.__init__(self, ob_space, net)
        self.x_mean = Variable(torch.zeros(1))
        self.x_std = Variable(torch.ones(1))

    def forward(self, obs):
        return self.net(obs).view(-1) * self.x_std + self.x_mean

    def set_mean(self, mean):
        self.x_mean.data.copy_(mean)

    def set_std(self, std):
        self.x_std.data.copy_(std)



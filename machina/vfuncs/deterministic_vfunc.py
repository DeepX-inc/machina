from .base import BaseVfunc
from ..utils import gpu_id

class DeterministicVfunc(BaseVfunc):
    def __init__(self, ob_space, net):
        BaseVfunc.__init__(self, ob_space)
        self.net = net
        if gpu_id != -1:
            self.cuda(gpu_id)

    def forward(self, obs):
        return self.net(obs).view(-1)

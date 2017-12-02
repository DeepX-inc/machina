from machina.vfuncs.base import BaseVfunc
from machina.utils import get_gpu

class DeterministicVfunc(BaseVfunc):
    def __init__(self, ob_space, net):
        BaseVfunc.__init__(self, ob_space)
        self.net = net
        gpu_id = get_gpu()
        if gpu_id != -1:
            self.cuda(gpu_id)

    def forward(self, obs):
        return self.net(obs).view(-1)

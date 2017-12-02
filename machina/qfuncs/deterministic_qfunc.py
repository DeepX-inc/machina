from machina.qfuncs.base import BaseQfunc
from machina.utils import get_gpu


class DeterministicQfunc(BaseQfunc):
    def __init__(self, ob_space, ac_space, net):
        BaseQfunc.__init__(self, ob_space, ac_space)
        self.net = net
        gpu_id = get_gpu()
        if gpu_id != -1:
            self.cuda(gpu_id)

    def forward(self, obs, acs):
        return self.net(obs, acs).view(-1)


from .base import BaseQfunc
from ..utils import gpu_id


class DeterministicQfunc(BaseQfunc):
    def __init__(self, ob_space, ac_space, net):
        BaseQfunc.__init__(self, ob_space, ac_space)
        self.net = net
        if gpu_id != -1:
            self.cuda(gpu_id)

    def forward(self, obs, acs):
        return self.net(obs, acs).view(-1)


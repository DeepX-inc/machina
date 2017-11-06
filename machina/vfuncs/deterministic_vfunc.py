
from .base import BaseVfunc

class DeterministicVfunc(BaseVfunc):
    def __init__(self, ob_space, net):
        BaseVfunc.__init__(self, ob_space)
        self.net = net

    def forward(self, obs):
        return self.net(obs).view(-1)


from .base import BaseQfunc

class DeterministicQfunc(BaseQfunc):
    def __init__(self, ob_space, ac_space, net):
        BaseQfunc.__init__(self, ob_space, ac_space)
        self.net = net

    def forward(self, obs, acs):
        return self.net(obs, acs).view(-1)


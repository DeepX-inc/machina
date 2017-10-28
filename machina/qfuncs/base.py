
import torch.nn as nn
class BaseQfunc(nn.Module):
    def __init__(self, ob_space, ac_space):
        nn.Module.__init__(self)
        self.ob_space = ob_space
        self.ac_space = ac_space


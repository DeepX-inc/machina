
import torch.nn as nn
class BaseVfunc(nn.Module):
    def __init__(self, ob_space):
        nn.Module.__init__(self)
        self.ob_space = ob_space

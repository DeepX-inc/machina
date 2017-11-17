import torch
import torch.nn as nn
from torch.autograd import Variable

class BasePol(nn.Module):
    def __init__(self, ob_space, ac_space, normalize_ac=True):
        nn.Module.__init__(self)
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.normalize_ac = normalize_ac

    def reset(self):
        pass


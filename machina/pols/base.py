import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class BasePol(nn.Module):
    def __init__(self, ob_space, ac_space, normalize_ac=True):
        nn.Module.__init__(self)
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.normalize_ac = normalize_ac

    def convert_ac_for_real(self, x):
        lb, ub = self.ac_space.low, self.ac_space.high
        if self.normalize_ac:
            x = lb + (x + 1.) * 0.5 * (ub - lb)
            x = np.clip(x, lb, ub)
        else:
            x = np.clip(x, lb, ub)
        return x



import numpy as np
import torch
from torch.autograd import Variable
import scipy

def discount_cumsum(x, discount):
    # See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
    # Here, we have y[t] - discount*y[t+1] = x[t]
    # or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    return scipy.signal.lfilter([1], [1, float(-discount)], x.numpy()[::-1], axis=0)[::-1]

class BaseData(object):
    pass

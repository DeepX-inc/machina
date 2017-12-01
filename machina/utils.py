import contextlib

import torch
import torch.autograd as autograd

from .misc import logger

# default gpu_id is -1.
# this means using cpu
gpu_id = -1

def set_gpu(device_id):
    global gpu_id
    gpu_id = device_id


@contextlib.contextmanager
def cpu_mode():
    """
    contextmanager
    set gpu_id to -1 while cpu_mode
    """
    global gpu_id
    _gpu_id = gpu_id
    gpu_id = -1
    yield
    gpu_id = _gpu_id


@contextlib.contextmanager
def measure(name):
    import time
    s = time.time()
    yield
    e = time.time()
    logger.log("{}: {:.4f}sec".format(name, e-s))


def torch2torch(tensor):
    """
    torch tensor with wrapping cuda
    """
    if gpu_id != -1:
        return tensor.cuda(gpu_id)
    else:
        return tensor


def np2torch(ndarray):
    """
    transform ndarray to torch tensor with cuda
    """
    return torch2torch(torch.from_numpy(ndarray))


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if gpu_id != -1:
            data = data.cuda(gpu_id)
        super(Variable, self).__init__(data, *args, **kwargs)



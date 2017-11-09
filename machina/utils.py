import contextlib
import torch.autograd as autograd

# default gpu_id is -1.
# this means using cpu
gpu_id = -1

def set_gpu(device_id):
    global gpu_id
    gpu_id = device_id


@contextlib.contextmanager
def cpu_mode():
    global gpu_id
    _gpu_id = gpu_id
    gpu_id = -1
    yield
    gpu_id = _gpu_id


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if gpu_id != -1:
            data = data.cuda(gpu_id)
        super(Variable, self).__init__(data, *args, **kwargs)


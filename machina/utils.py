# Copyright 2018 DeepX Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import contextlib

import torch
import torch.autograd as autograd

from machina.misc import logger

_DEVICE = torch.device('cpu')

def set_device(device):
    global _DEVICE
    _DEVICE = device


def get_device():
    return _DEVICE


@contextlib.contextmanager
def cpu_mode():
    global _DEVICE
    tmp = _DEVICE
    _DEVICE = torch.device('cpu')
    yield
    _DEVICE = tmp


@contextlib.contextmanager
def measure(name):
    import time
    s = time.time()
    yield
    e = time.time()
    logger.log("{}: {:.4f}sec".format(name, e-s))

<<<<<<< HEAD

def torch2torch(tensor):
    """
    torch tensor with wrapping cuda
    """
    if _GPU_ID != -1:
        return tensor.cuda(_GPU_ID)
    else:
        return tensor


def np2torch(ndarray):
    """
    transform ndarray to torch tensor with cuda
    """
    return torch2torch(torch.from_numpy(ndarray))


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        data = torch2torch(data)
        super(Variable, self).__init__(data, *args, **kwargs)
=======
>>>>>>> b773ba33725bec9495abbb52a3b3fa2af045081b

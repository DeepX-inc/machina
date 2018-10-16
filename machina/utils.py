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
    if device < 0:
        _DEVICE = 'cpu'
    else:
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

def detach_tensor_dict(d):
    _d = dict()
    for key in d.keys():
        if d[key] is None:
            continue
        if isinstance(d[key], tuple):
            _d[key] = (d[key][0].detach(), d[key][1].detach())
            continue
        _d[key] = d[key].detach()
    return _d


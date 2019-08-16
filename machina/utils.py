import contextlib

import redis
import torch
import torch.autograd as autograd

from machina import logger

_DEVICE = torch.device('cpu')

_REDIS = None


def make_redis(redis_host, redis_port):
    r = redis.StrictRedis(redis_host, redis_port)
    set_redis(r)


def set_redis(r):
    global _REDIS
    _REDIS = r


def get_redis():
    return _REDIS


def _int(v):
    try:
        new_v = int(v)
    except:
        new_v = -1
    return new_v


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
def measure(name, log_enable=True):
    import time
    s = time.time()
    yield
    e = time.time()
    if log_enable:
        logger.log("{}: {:.4f}sec".format(name, e - s))


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


def wrap_ddp(cls):
    """Return wrapper class for the torch.DDP and apex. Delegete getattr to the
    inner module.
    """
    class _Wrap(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getattr__(self, name):
            wrapped_module = super().__getattr__('module')
            if hasattr(wrapped_module, name):
                return getattr(wrapped_module, name)
            return super().__getattr__(name)
    return _Wrap


def init_ray(num_cpus=None, num_gpus=None, ray_redis_address=None):
    """Initialize ray. If `ray_redis_address` is given, use the address to
    connect existing ray cluster. Otherwise start ray locally.
    """
    import ray
    if ray_redis_address is not None:
        ray.init(redis_address=ray_redis_address)
    else:
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        ray.init(num_gpus=num_gpus, num_cpus=num_cpus)

    # XXX: Currently, ray (pyarrow) does not serialize `requires_grad`
    # attribute. As a workaround, use custom serializer.
    # See https://github.com/ray-project/ray/issues/4855
    ray.register_custom_serializer(torch.nn.Module, use_pickle=True)

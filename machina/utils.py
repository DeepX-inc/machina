import contextlib
import copy

import numpy as np
import redis
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.autograd as autograd
try:
    import ray
except ImportError:
    pass

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


def state_dict_to_cpu(state_dict):
    """Converting state dict with cpu version
    """
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            state_dict[k] = v.to("cpu")
        if isinstance(v, dict):
            state_dict[k] = state_dict_to_cpu(v)
    return state_dict


def get_cpu_state_dict(module):
    return state_dict_to_cpu(copy.deepcopy(module.state_dict()))


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


def make_model_distributed(model, optim,
                           use_apex=False,
                           apex_opt_level="O0",
                           apex_keep_batchnorm_fp32=True,
                           apex_sync_bn=False,
                           apex_loss_scale=None,
                           device_ids=None, output_device=None,
                           ):
    """Return model for distributed trainings.
    Note that returned model shares parameters with the original model.
    """
    if use_apex:
        global amp
        global apex
        import apex.parallel
        from apex import amp
        ddp_model, optim = amp.initialize(model, optim,
                                          opt_level=apex_opt_level,
                                          keep_batchnorm_fp32=apex_keep_batchnorm_fp32,
                                          loss_scale=apex_loss_scale)
        ddp_cls = wrap_ddp(apex.parallel.DistributedDataParallel)
        ddp_model = ddp_cls(ddp_model)
        if apex_sync_bn:
            ddp_model = apex.parallel.convert_syncbn_model(model)
    else:
        ddp_cls = wrap_ddp(nn.parallel.DistributedDataParallel)
        ddp_model = ddp_cls(model, device_ids, output_device)

    return ddp_model, optim


def init_ray(num_cpus=None, num_gpus=None, ray_redis_address=None):
    """Initialize ray. If `ray_redis_address` is given, use the address to
    connect existing ray cluster. Otherwise start ray locally.
    """
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


class BaseDistributedRayTrainer(object):
    """Base class for multi-GPU trainings using ray
    """
    @classmethod
    def as_remote(cls, num_gpus=1, resources=None):
        return ray.remote(num_cpus=0, num_gpus=num_gpus, resources=resources)(cls)

    def __init__(self, rank=0, world_size=1, master_address=None, backend="nccl", seed=1234):
        dist.init_process_group(
            backend=backend, init_method=master_address, world_size=world_size, rank=rank)

        # Ray automatically sets CUDA_VISIBLE_DEVICES. Use cuda:0
        self.device = "cuda:0"
        np.random.seed(seed)
        torch.manual_seed(seed)
        set_device(self.device)
        torch.cuda.set_device(0)

    def get_state(self, name):
        """Get `name`'s state_dict in this class
        """
        return state_dict_to_cpu(copy.deepcopy(getattr(self, name).state_dict()))


class TrainManager(object):
    """Manage one or multiple trainer(s)
    """

    def __init__(self, trainer_cls, num_trainer=1, master_address=None,
                 **kwargs):
        if not ray.is_initialized():
            init_ray()

        self.trainers = [trainer_cls.as_remote().remote(**kwargs,
                                                        rank=i,
                                                        world_size=num_trainer,
                                                        master_address=master_address)
                         for i in range(num_trainer)]

    def get_state(self, name):
        """Get state of instance of nn.Module
        """
        state = self.trainers[0].get_state.remote(name)
        state = ray.get(state)
        return state

    def train(self, **kwargs):
        args = {}
        for k, v in kwargs.items():
            args[k] = ray.put(v)
        results = [t.train.remote(**args) for t in self.trainers]
        results = ray.get(results)
        # only return a result of the master node
        # (because all trainers have same result)
        return results[0]

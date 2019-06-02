"""
Sampler class for multi node situation.
This sampler uses redis for the backend.
"""

import argparse
import time

import cloudpickle
import redis

from machina.samplers import EpiSampler
from machina.utils import _int, get_redis, make_redis


class DistributedEpiSampler(object):
    """
    A sampler which sample episodes.

    Parameters
    ----------
    world_size : int
        Number of nodes
    rank : int
        -1 represent master node.
    env : gym.Env
    pol : Pol
    num_parallel : int
        Number of processes
    prepro : Prepro
    seed : int
    """

    def __init__(self, world_size, rank=-1, env=None, pol=None, num_parallel=8, prepro=None, seed=256, flush_db=False):
        if rank < 0:
            assert env is not None and pol is not None

        self.world_size = world_size
        self.rank = rank

        self.r = get_redis()

        if flush_db:
            # reset DB
            keys = self.r.keys(pattern="*_trigger_*")
            if keys:
                self.r.delete(*keys)

        if rank < 0:
            self.env = env
            self.pol = pol
            self.num_parallel = num_parallel // world_size
            self.prepro = prepro
            self.seed = seed

            self.original_num_parallel = num_parallel

        self.scatter_from_master('env')
        self.scatter_from_master('pol')
        self.scatter_from_master('num_parallel')
        self.scatter_from_master('prepro')
        self.scatter_from_master('seed')

        self.seed = self.seed * (self.rank + 23000)

        if not rank < 0:
            self.in_node_sampler = EpiSampler(
                self.env, self.pol, self.num_parallel, self.prepro, self.seed)
            self.launch_sampler()

    def __del__(self):
        if not self.rank < 0:
            del self.in_node_sampler

    def launch_sampler(self):
        while True:
            self.scatter_from_master('pol')
            self.scatter_from_master('max_epis')
            self.scatter_from_master('max_steps')
            self.scatter_from_master('deterministic')

            self.epis = self.in_node_sampler.sample(
                self.pol, self.max_epis, self.max_steps, self.deterministic)

            self.gather_to_master('epis')

    def sync(self, keys, target_value):
        """Wait until all `keys` become `target_value`
        """
        while True:
            values = self.r.mget(keys)
            if all([_int(v) == target_value for v in values]):
                break
            time.sleep(0.1)

    def wait_trigger(self, trigger):
        """Wait until `trigger` become 1
        """
        self.sync(trigger, 1)

    def wait_trigger_processed(self, trigger):
        """Wait until `trigger` become 0
        """
        self.sync(trigger, 0)

    def set_trigger(self, trigger, value='1'):
        """Set all triggers to `value`
        """
        if not isinstance(trigger, (list, tuple)):
            trigger = [trigger]
        mapping = {k: value for k in trigger}
        self.r.mset(mapping)

    def reset_trigger(self, trigger):
        """Set all triggers to 0
        """
        self.set_trigger(trigger, value='0')

    def wait_trigger_completion(self, trigger):
        """Set trigger to 1, then wait until it become 0
        """
        self.set_trigger(trigger)
        self.wait_trigger_processed(trigger)

    def scatter_from_master(self, key):
        """
        master: set `key` to DB, then set trigger and wait sampler completion
        sampler: wait trigger, then get `key` and reset trigger
        """

        if self.rank < 0:
            obj = getattr(self, key)
            self.r.set(key, cloudpickle.dumps(obj))
            trigger = ['{}_trigger_{}'.format(
                key, rank) for rank in range(self.world_size)]
            self.wait_trigger_completion(trigger)
        else:
            trigger = '{}_trigger_{}'.format(key, self.rank)
            self.wait_trigger(trigger)
            obj = cloudpickle.loads(self.r.get(key))
            setattr(self, key, obj)
            self.reset_trigger(trigger)

    def gather_to_master(self, key):
        """
        master: wait trigger, then get the value from DB
        sampler: set `key` to DB, then wait master fetch

        This method assume that obj is summable to list.
        """

        if self.rank < 0:
            num_done = 0
            objs = []
            while True:
                time.sleep(0.1)
                # This for iteration can be faster.
                for rank in range(self.world_size):
                    trigger = self.r.get(key + '_trigger' + "_{}".format(rank))
                    if _int(trigger) == 1:
                        obj = cloudpickle.loads(
                            self.r.get(key + "_{}".format(rank)))
                        objs += obj
                        self.r.set(key + '_trigger' + "_{}".format(rank), '0')
                        num_done += 1
                if num_done == self.world_size:
                    break
            setattr(self, key, objs)
        else:
            obj = getattr(self, key)
            self.r.set(key + "_{}".format(self.rank), cloudpickle.dumps(obj))
            trigger = '{}_trigger_{}'.format(key, self.rank)
            self.wait_trigger_completion(trigger)

    def sample(self, pol, max_epis=None, max_steps=None, deterministic=False):
        """
        This method should be called in master node.
        """
        self.pol = pol
        self.max_epis = max_epis // self.world_size if max_epis is not None else None
        self.max_steps = max_steps // self.world_size if max_steps is not None else None
        self.deterministic = deterministic

        self.scatter_from_master('pol')
        self.scatter_from_master('max_epis')
        self.scatter_from_master('max_steps')
        self.scatter_from_master('deterministic')

        self.gather_to_master('epis')

        return self.epis


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--redis_host', type=str, default='localhost')
    parser.add_argument('--redis_port', type=str, default='6379')
    parser.add_argument('--flush_db', action="store_true")
    args = parser.parse_args()

    make_redis(args.redis_host, args.redis_port)

    sampler = DistributedEpiSampler(
        args.world_size, args.rank, flush_db=args.flush_db)

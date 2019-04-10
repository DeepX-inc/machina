import os
from signal import SIGTERM
import subprocess
import unittest

import numpy as np
import psutil

from machina.traj import Traj
from machina.envs import GymEnv
from machina.samplers import EpiSampler, DistributedEpiSampler
from machina.pols.random_pol import RandomPol


class TestTraj(unittest.TestCase):

    env = None
    traj = None

    @classmethod
    def setUpClass(cls):
        cls.env = GymEnv('Pendulum-v0')
        cls.pol = RandomPol(cls.env.observation_space, cls.env.action_space)

    def test_epi_sampler(self):
        sampler = EpiSampler(self.env, self.pol, num_parallel=1)
        epis = sampler.sample(self.pol, max_epis=2)
        assert len(epis) >= 2

    def test_distributed_epi_sampler(self):
        proc_redis = subprocess.Popen(['redis-server'])
        proc_slave = subprocess.Popen(['python', '-m', 'machina.samplers.distributed_epi_sampler', '--world_size', '1', '--rank', '0', '--redis_host', 'localhost', '--redis_port', '6379'])
        sampler = DistributedEpiSampler(1, -1, 'localhost', '6379', self.env, self.pol, num_parallel=1)
        epis = sampler.sample(self.pol, max_epis=2)
        assert len(epis) >= 2
        children = psutil.Process(os.getpid()).children(recursive=True)
        for child in children:
            child.send_signal(SIGTERM)

if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np

from machina.traj import Traj
from machina.envs import GymEnv
from machina.samplers import EpiSampler
from machina.pols.random_pol import RandomPol


class TestTraj(unittest.TestCase):

    env = None
    traj = None

    @classmethod
    def setUpClass(cls):
        cls.env = GymEnv('Pendulum-v0')
        pol = RandomPol(cls.env.observation_space, cls.env.action_space)
        sampler = EpiSampler(cls.env, pol, num_parallel=1)
        epis = sampler.sample(pol, max_steps=32)

        cls.traj = Traj()
        cls.traj.add_epis(epis)
        cls.traj.register_epis()

    def test_add_traj(self):
        new_traj = Traj()
        new_traj.add_traj(self.traj)
        assert new_traj.num_epi == self.traj.num_epi
        assert new_traj.num_step == self.traj.num_step

    def test_random_batch_once(self):
        batch_size = 32
        data_map = self.traj.random_batch_once(
            batch_size, return_indices=False)

        data_map, indices = self.traj.random_batch_once(
            batch_size, return_indices=True)

        data_map = self.traj.random_batch_once(
            batch_size, indices=np.arange(5), return_indices=False)

        data_map, indices = self.traj.random_batch_once(
            batch_size, indices=np.arange(5), return_indices=True)

    def test_random_batch(self):
        batch_size = 32
        iterator = self.traj.random_batch(batch_size)

        iterator = self.traj.random_batch(batch_size, return_indices=False)
        for batch in iterator:
            pass

        iterator = self.traj.random_batch(batch_size, return_indices=True)
        for batch, indices in iterator:
            pass

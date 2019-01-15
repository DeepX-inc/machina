"""
Test script for environment
"""

import unittest

import gym
import numpy as np
import torch

from machina.envs import GymEnv, C2DEnv


def test_continuous2discrete():
    continuous_env = GymEnv('Pendulum-v0', record_video=False)
    discrete_env = C2DEnv(continuous_env, n_bins=10)

    assert np.all(discrete_env.action_space.nvec == np.array([10]))

    discrete_env.reset()
    out = discrete_env.step([3, 10])


if __name__ == '__main__':
    test_continuous2discrete()

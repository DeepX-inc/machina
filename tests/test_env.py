"""
Test script for environment
"""

import unittest

import gym
import numpy as np
import torch
from torch import nn

from gym.wrappers import FlattenDictWrapper
from machina.envs import GymEnv, C2DEnv, flatten_to_dict
from simple_net import PolDictNet, VNet, QNet
from machina.vfuncs import DeterministicSVfunc, DeterministicSAVfunc
from machina.pols import GaussianPol
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina.algos import ppo_clip, sac
from gym.envs import register

register(
    id='PendulumDictEnv-v0',
    entry_point='tests.env:PendulumDictEnv',
    max_episode_steps=200
)


def test_continuous2discrete():
    continuous_env = GymEnv('Pendulum-v0', record_video=False)
    discrete_env = C2DEnv(continuous_env, n_bins=10)

    assert np.all(discrete_env.action_space.nvec == np.array([10]))

    discrete_env.reset()
    out = discrete_env.step([3, 10])


def test_flatten2dict():
    dict_env = gym.make('PendulumDictEnv-v0')
    dict_env = GymEnv(dict_env)
    dict_ob = dict_env.ob_space.sample()
    dict_ob_space = dict_env.ob_space
    env = FlattenDictWrapper(
        dict_env, dict_env.ob_space.spaces.keys())
    flatten_ob = env.observation(dict_ob)
    dict_keys = env.dict_keys
    recovered_dict_ob = flatten_to_dict(
        flatten_ob, dict_ob_space, dict_keys)
    tf = []
    for (a_key, a_val), (b_key, b_val) in zip(dict_ob.items(), recovered_dict_ob.items()):
        tf.append(a_key == b_key)
        tf.append(all(a_val == b_val))
    assert all(tf)


class TestFlatten2DictPP0(unittest.TestCase):
    def setUp(self):
        dict_env = gym.make('PendulumDictEnv-v0')
        self.dict_ob_space = dict_env.observation_space
        env = FlattenDictWrapper(
            dict_env, dict_env.observation_space.spaces.keys())
        self.env = GymEnv(env)

    def test_learning(self):
        pol_net = PolDictNet(self.dict_ob_space,
                             self.env.ac_space, h1=32, h2=32)
        pol = GaussianPol(self.env.ob_space,
                          self.env.ac_space, pol_net)

        vf_net = VNet(self.env.ob_space, h1=32, h2=32)
        vf = DeterministicSVfunc(self.env.ob_space, vf_net)

        sampler = EpiSampler(self.env, pol, num_parallel=1)

        optim_pol = torch.optim.Adam(pol_net.parameters(), 3e-4)
        optim_vf = torch.optim.Adam(vf_net.parameters(), 3e-4)

        epis = sampler.sample(pol, max_steps=32)

        traj = Traj()
        traj.add_epis(epis)

        traj = ef.compute_vs(traj, vf)
        traj = ef.compute_rets(traj, 0.99)
        traj = ef.compute_advs(traj, 0.99, 0.95)
        traj = ef.centerize_advs(traj)
        traj = ef.compute_h_masks(traj)
        traj.register_epis()

        result_dict = ppo_clip.train(traj=traj, pol=pol, vf=vf, clip_param=0.2,
                                     optim_pol=optim_pol, optim_vf=optim_vf, epoch=1, batch_size=32)

        del sampler


class TestFlatten2DictSAC(unittest.TestCase):
    def setUp(self):
        dict_env = gym.make('PendulumDictEnv-v0')
        self.dict_ob_space = dict_env.observation_space
        env = FlattenDictWrapper(
            dict_env, dict_env.observation_space.spaces.keys())
        self.env = GymEnv(env)

    def test_learning(self):
        pol_net = PolDictNet(self.dict_ob_space,
                             self.env.ac_space, h1=32, h2=32)
        pol = GaussianPol(self.env.ob_space, self.env.ac_space, pol_net)

        qf_net1 = QNet(self.env.ob_space, self.env.ac_space)
        qf1 = DeterministicSAVfunc(
            self.env.ob_space, self.env.ac_space, qf_net1)
        targ_qf_net1 = QNet(self.env.ob_space, self.env.ac_space)
        targ_qf_net1.load_state_dict(qf_net1.state_dict())
        targ_qf1 = DeterministicSAVfunc(
            self.env.ob_space, self.env.ac_space, targ_qf_net1)

        qf_net2 = QNet(self.env.ob_space, self.env.ac_space)
        qf2 = DeterministicSAVfunc(
            self.env.ob_space, self.env.ac_space, qf_net2)
        targ_qf_net2 = QNet(self.env.ob_space, self.env.ac_space)
        targ_qf_net2.load_state_dict(qf_net2.state_dict())
        targ_qf2 = DeterministicSAVfunc(
            self.env.ob_space, self.env.ac_space, targ_qf_net2)

        qfs = [qf1, qf2]
        targ_qfs = [targ_qf1, targ_qf2]

        log_alpha = nn.Parameter(torch.zeros(()))

        sampler = EpiSampler(self.env, pol, num_parallel=1)

        optim_pol = torch.optim.Adam(pol_net.parameters(), 3e-4)
        optim_qf1 = torch.optim.Adam(qf_net1.parameters(), 3e-4)
        optim_qf2 = torch.optim.Adam(qf_net2.parameters(), 3e-4)
        optim_qfs = [optim_qf1, optim_qf2]
        optim_alpha = torch.optim.Adam([log_alpha], 3e-4)

        epis = sampler.sample(pol, max_steps=32)

        traj = Traj()
        traj.add_epis(epis)

        traj = ef.add_next_obs(traj)
        traj.register_epis()

        result_dict = sac.train(
            traj,
            pol, qfs, targ_qfs, log_alpha,
            optim_pol, optim_qfs, optim_alpha,
            2, 32,
            0.01, 0.99, 2,
        )

        del sampler


if __name__ == '__main__':
    test_continuous2discrete()

"""
Test script for environment
"""

import unittest

import gym
import numpy as np
import torch
from torch import nn

try:
    from gym.wrappers import FlattenDictWrapper
except:
    # gym 0.15.4 remove FlattendDictWrapper
    from gym.wrappers import FilterObservation, FlattenObservation
from machina.envs import GymEnv, C2DEnv, flatten_to_dict
from simple_net import PolDictNet, VNet, QNet, VNetLSTM, PolNetDictLSTM, QNetLSTM
from machina.vfuncs import DeterministicSVfunc, DeterministicSAVfunc
from machina.pols import GaussianPol
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina.algos import ppo_clip, sac, r2d2_sac
from gym.envs import register

register(
    id='PendulumDictEnv-v0',
    entry_point='tests.env:PendulumDictEnv',
    max_episode_steps=200
)


def _make_flat(*args, **kargs):
    if "FlattenDictWrapper" in dir():
        return FlattenDictWrapper(*args, **kargs)
    return FlattenObservation(FilterObservation(*args, **kargs))


def test_continuous2discrete():
    continuous_env = GymEnv('Pendulum-v0', record_video=False)
    discrete_env = C2DEnv(continuous_env, n_bins=10)

    assert np.all(discrete_env.action_space.nvec == np.array([10]))

    discrete_env.reset()
    out = discrete_env.step([3, 10])


def test_flatten2dict():
    dict_env = gym.make('PendulumDictEnv-v0')
    dict_env = GymEnv(dict_env)
    dict_ob = dict_env.observation_space.sample()
    dict_observation_space = dict_env.observation_space
    dict_keys = dict_env.observation_space.spaces.keys()
    env = _make_flat(dict_env, dict_keys)
    flatten_ob = env.observation(dict_ob)
    recovered_dict_ob = flatten_to_dict(
        flatten_ob, dict_observation_space, dict_keys)
    tf = []
    for (a_key, a_val), (b_key, b_val) in zip(dict_ob.items(), recovered_dict_ob.items()):
        tf.append(a_key == b_key)
        tf.append(all(a_val == b_val))
    assert all(tf)


class TestFlatten2DictPP0(unittest.TestCase):
    def setUp(self):
        dict_env = gym.make('PendulumDictEnv-v0')
        self.dict_observation_space = dict_env.observation_space
        env = _make_flat(
            dict_env, dict_env.observation_space.spaces.keys())
        self.env = GymEnv(env)

    def test_learning(self):
        pol_net = PolDictNet(self.dict_observation_space,
                             self.env.action_space, h1=32, h2=32)
        pol = GaussianPol(self.env.observation_space,
                          self.env.action_space, pol_net)

        vf_net = VNet(self.env.observation_space, h1=32, h2=32)
        vf = DeterministicSVfunc(self.env.observation_space, vf_net)

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

    def test_learning_rnn(self):
        pol_net = PolNetDictLSTM(
            self.dict_observation_space, self.env.action_space, h_size=32, cell_size=32)
        pol = GaussianPol(self.env.observation_space,
                          self.env.action_space, pol_net, rnn=True)

        vf_net = VNetLSTM(self.env.observation_space, h_size=32, cell_size=32)
        vf = DeterministicSVfunc(self.env.observation_space, vf_net, rnn=True)

        sampler = EpiSampler(self.env, pol, num_parallel=1)

        optim_pol = torch.optim.Adam(pol_net.parameters(), 3e-4)
        optim_vf = torch.optim.Adam(vf_net.parameters(), 3e-4)

        epis = sampler.sample(pol, max_steps=400)

        traj = Traj()
        traj.add_epis(epis)

        traj = ef.compute_vs(traj, vf)
        traj = ef.compute_rets(traj, 0.99)
        traj = ef.compute_advs(traj, 0.99, 0.95)
        traj = ef.centerize_advs(traj)
        traj = ef.compute_h_masks(traj)
        traj.register_epis()

        result_dict = ppo_clip.train(traj=traj, pol=pol, vf=vf, clip_param=0.2,
                                     optim_pol=optim_pol, optim_vf=optim_vf, epoch=1, batch_size=2)

        del sampler


class TestFlatten2DictSAC(unittest.TestCase):
    def setUp(self):
        dict_env = gym.make('PendulumDictEnv-v0')
        self.dict_observation_space = dict_env.observation_space
        env = _make_flat(
            dict_env, dict_env.observation_space.spaces.keys())
        self.env = GymEnv(env)

    def test_learning(self):
        pol_net = PolDictNet(self.dict_observation_space,
                             self.env.action_space, h1=32, h2=32)
        pol = GaussianPol(self.env.observation_space,
                          self.env.action_space, pol_net)

        qf_net1 = QNet(self.env.observation_space, self.env.action_space)
        qf1 = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, qf_net1)
        targ_qf_net1 = QNet(self.env.observation_space, self.env.action_space)
        targ_qf_net1.load_state_dict(qf_net1.state_dict())
        targ_qf1 = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, targ_qf_net1)

        qf_net2 = QNet(self.env.observation_space, self.env.action_space)
        qf2 = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, qf_net2)
        targ_qf_net2 = QNet(self.env.observation_space, self.env.action_space)
        targ_qf_net2.load_state_dict(qf_net2.state_dict())
        targ_qf2 = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, targ_qf_net2)

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


class TestFlatten2DictR2D2SAC(unittest.TestCase):
    def setUp(self):
        dict_env = gym.make('PendulumDictEnv-v0')
        self.dict_observation_space = dict_env.observation_space
        env = _make_flat(
            dict_env, dict_env.observation_space.spaces.keys())
        self.env = GymEnv(env)

    def test_learning(self):
        pol_net = PolNetDictLSTM(
            self.dict_observation_space, self.env.action_space, h_size=32, cell_size=32)
        pol = GaussianPol(self.env.observation_space,
                          self.env.action_space, pol_net, rnn=True)

        qf_net1 = QNetLSTM(self.env.observation_space,
                           self.env.action_space, h_size=32, cell_size=32)
        qf1 = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, qf_net1, rnn=True)
        targ_qf_net1 = QNetLSTM(
            self.env.observation_space, self.env.action_space, h_size=32, cell_size=32)
        targ_qf_net1.load_state_dict(qf_net1.state_dict())
        targ_qf1 = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, targ_qf_net1, rnn=True)

        qf_net2 = QNetLSTM(self.env.observation_space,
                           self.env.action_space, h_size=32, cell_size=32)
        qf2 = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, qf_net2, rnn=True)
        targ_qf_net2 = QNetLSTM(
            self.env.observation_space, self.env.action_space, h_size=32, cell_size=32)
        targ_qf_net2.load_state_dict(qf_net2.state_dict())
        targ_qf2 = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, targ_qf_net2, rnn=True)

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
        max_pri = traj.get_max_pri()
        traj = ef.set_all_pris(traj, max_pri)
        traj = ef.compute_seq_pris(traj, 4)
        traj = ef.compute_h_masks(traj)
        for i in range(len(qfs)):
            traj = ef.compute_hs(
                traj, qfs[i], hs_name='q_hs'+str(i), input_acs=True)
            traj = ef.compute_hs(
                traj, targ_qfs[i], hs_name='targ_q_hs'+str(i), input_acs=True)
        traj.register_epis()

        result_dict = r2d2_sac.train(
            traj,
            pol, qfs, targ_qfs, log_alpha,
            optim_pol, optim_qfs, optim_alpha,
            2, 32, 4, 2,
            0.01, 0.99, 2,
        )

        del sampler


if __name__ == '__main__':
    test_continuous2discrete()

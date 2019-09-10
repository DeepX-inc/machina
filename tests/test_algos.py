"""
Test script for algorithms.
"""

import unittest
import os

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import machina as mc
from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol
from machina.pols import DeterministicActionNoisePol, ArgmaxQfPol, MPCPol, RandomPol
from machina.noise import OUActionNoise
from machina.algos import ppo_clip, ppo_kl, trpo, ddpg, sac, svg, qtopt, on_pol_teacher_distill, behavior_clone, gail, airl, mpc, r2d2_sac, diayn, diayn_sac
from machina.vfuncs import DeterministicSVfunc, DeterministicSAVfunc, CEMDeterministicSAVfunc
from machina.models import DeterministicSModel
from machina.envs import GymEnv, C2DEnv, SkillEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import measure, set_device
from simple_net import PolNet, VNet, ModelNet, PolNetLSTM, VNetLSTM, ModelNetLSTM, QNet, QNetLSTM, DiscrimNet, DiaynDiscrimNet


class TestPPOContinuous(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('Pendulum-v0')

    def test_learning(self):
        pol_net = PolNet(self.env.observation_space,
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
        result_dict = ppo_kl.train(traj=traj, pol=pol, vf=vf, kl_beta=0.1, kl_targ=0.2,
                                   optim_pol=optim_pol, optim_vf=optim_vf, epoch=1, batch_size=32, max_grad_norm=10)

        del sampler

    def test_learning_rnn(self):
        pol_net = PolNetLSTM(
            self.env.observation_space, self.env.action_space, h_size=32, cell_size=32)
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
        result_dict = ppo_kl.train(traj=traj, pol=pol, vf=vf, kl_beta=0.1, kl_targ=0.2,
                                   optim_pol=optim_pol, optim_vf=optim_vf, epoch=1, batch_size=2, max_grad_norm=20)

        del sampler


class TestPPODiscrete(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('CartPole-v0')

    def test_learning(self):
        pol_net = PolNet(self.env.observation_space,
                         self.env.action_space, h1=32, h2=32)
        pol = CategoricalPol(self.env.observation_space,
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
        result_dict = ppo_kl.train(traj=traj, pol=pol, vf=vf, kl_beta=0.1, kl_targ=0.2,
                                   optim_pol=optim_pol, optim_vf=optim_vf, epoch=1, batch_size=32, max_grad_norm=10)

        del sampler

    # def test_learning_rnn(self):
    #    pol_net = PolNetLSTM(self.env.observation_space, self.env.action_space, h_size=32, cell_size=32)
    #    pol = CategoricalPol(self.env.observation_space, self.env.action_space, pol_net, rnn=True)

    #    vf_net = VNetLSTM(self.env.observation_space, h_size=32, cell_size=32)
    #    vf = DeterministicSVfunc(self.env.observation_space, vf_net, rnn=True)

    #    sampler = EpiSampler(self.env, pol, num_parallel=1)

    #    optim_pol = torch.optim.Adam(pol_net.parameters(), 3e-4)
    #    optim_vf = torch.optim.Adam(vf_net.parameters(), 3e-4)

    #    epis = sampler.sample(pol, max_steps=400)

    #    traj = Traj()
    #    traj.add_epis(epis)

    #    traj = ef.compute_vs(traj, vf)
    #    traj = ef.compute_rets(traj, 0.99)
    #    traj = ef.compute_advs(traj, 0.99, 0.95)
    #    traj = ef.centerize_advs(traj)
    #    traj = ef.compute_h_masks(traj)
    #    traj.register_epis()

    #    result_dict = ppo_clip.train(traj=traj, pol=pol, vf=vf, clip_param=0.2,
    #                                    optim_pol=optim_pol, optim_vf=optim_vf, epoch=1, batch_size=2)
    #    result_dict = ppo_kl.train(traj=traj, pol=pol, vf=vf, kl_beta=0.1, kl_targ=0.2,
    #                                optim_pol=optim_pol, optim_vf=optim_vf, epoch=1, batch_size=2, max_grad_norm=20)

    #    del sampler


class TestTRPOContinuous(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('Pendulum-v0')

    def test_learning(self):
        pol_net = PolNet(self.env.observation_space,
                         self.env.action_space, h1=32, h2=32)
        pol = GaussianPol(self.env.observation_space,
                          self.env.action_space, pol_net)

        vf_net = VNet(self.env.observation_space, h1=32, h2=32)
        vf = DeterministicSVfunc(self.env.observation_space, vf_net)

        sampler = EpiSampler(self.env, pol, num_parallel=1)

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

        result_dict = trpo.train(traj, pol, vf, optim_vf, 1, 24)

        del sampler

    def test_learning_rnn(self):
        pol_net = PolNetLSTM(
            self.env.observation_space, self.env.action_space, h_size=32, cell_size=32)
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

        result_dict = trpo.train(traj, pol, vf, optim_vf, 1, 2)

        del sampler


class TestTRPODiscrete(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('CartPole-v0')

    def test_learning(self):
        pol_net = PolNet(self.env.observation_space,
                         self.env.action_space, h1=32, h2=32)
        pol = CategoricalPol(self.env.observation_space,
                             self.env.action_space, pol_net)

        vf_net = VNet(self.env.observation_space, h1=32, h2=32)
        vf = DeterministicSVfunc(self.env.observation_space, vf_net)

        sampler = EpiSampler(self.env, pol, num_parallel=1)

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

        result_dict = trpo.train(traj, pol, vf, optim_vf, 1, 24)

        del sampler

    def test_learning_rnn(self):
        pol_net = PolNetLSTM(
            self.env.observation_space, self.env.action_space, h_size=32, cell_size=32)
        pol = CategoricalPol(
            self.env.observation_space, self.env.action_space, pol_net, rnn=True)

        vf_net = VNetLSTM(self.env.observation_space, h_size=32, cell_size=32)
        vf = DeterministicSVfunc(self.env.observation_space, vf_net, rnn=True)

        sampler = EpiSampler(self.env, pol, num_parallel=1)

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

        result_dict = trpo.train(traj, pol, vf, optim_vf, 1, 2)

        del sampler


class TestDDPG(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('Pendulum-v0')

    def test_learning(self):
        pol_net = PolNet(self.env.observation_space, self.env.action_space,
                         h1=32, h2=32, deterministic=True)
        noise = OUActionNoise(self.env.action_space)
        pol = DeterministicActionNoisePol(
            self.env.observation_space, self.env.action_space, pol_net, noise)

        targ_pol_net = PolNet(
            self.env.observation_space, self.env.action_space, 32, 32, deterministic=True)
        targ_pol_net.load_state_dict(pol_net.state_dict())
        targ_noise = OUActionNoise(self.env.action_space)
        targ_pol = DeterministicActionNoisePol(
            self.env.observation_space, self.env.action_space, targ_pol_net, targ_noise)

        qf_net = QNet(self.env.observation_space,
                      self.env.action_space, h1=32, h2=32)
        qf = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, qf_net)

        targ_qf_net = QNet(self.env.observation_space,
                           self.env.action_space, 32, 32)
        targ_qf_net.load_state_dict(targ_qf_net.state_dict())
        targ_qf = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, targ_qf_net)

        sampler = EpiSampler(self.env, pol, num_parallel=1)

        optim_pol = torch.optim.Adam(pol_net.parameters(), 3e-4)
        optim_qf = torch.optim.Adam(qf_net.parameters(), 3e-4)

        epis = sampler.sample(pol, max_steps=32)

        traj = Traj()
        traj.add_epis(epis)

        traj = ef.add_next_obs(traj)
        traj.register_epis()

        result_dict = ddpg.train(
            traj, pol, targ_pol, qf, targ_qf, optim_pol, optim_qf, 1, 32, 0.01, 0.9)

        del sampler


class TestSVG(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('Pendulum-v0')

    def test_learning(self):
        pol_net = PolNet(self.env.observation_space,
                         self.env.action_space, h1=32, h2=32)
        pol = GaussianPol(self.env.observation_space,
                          self.env.action_space, pol_net)

        targ_pol_net = PolNet(self.env.observation_space,
                              self.env.action_space, 32, 32)
        targ_pol_net.load_state_dict(pol_net.state_dict())
        targ_pol = GaussianPol(
            self.env.observation_space, self.env.action_space, targ_pol_net)

        qf_net = QNet(self.env.observation_space,
                      self.env.action_space, h1=32, h2=32)
        qf = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, qf_net)

        targ_qf_net = QNet(self.env.observation_space,
                           self.env.action_space, 32, 32)
        targ_qf_net.load_state_dict(targ_qf_net.state_dict())
        targ_qf = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, targ_qf_net)

        sampler = EpiSampler(self.env, pol, num_parallel=1)

        optim_pol = torch.optim.Adam(pol_net.parameters(), 3e-4)
        optim_qf = torch.optim.Adam(qf_net.parameters(), 3e-4)

        epis = sampler.sample(pol, max_steps=32)

        traj = Traj()
        traj.add_epis(epis)

        traj = ef.add_next_obs(traj)
        traj.register_epis()

        result_dict = svg.train(
            traj, pol, targ_pol, qf, targ_qf, optim_pol, optim_qf, 1, 32, 0.01, 0.9, 1)

        del sampler


class TestSAC(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('Pendulum-v0')

    def test_learning(self):
        pol_net = PolNet(self.env.observation_space,
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


class TestQTOPT(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('Pendulum-v0')

    def test_learning(self):
        qf_net = QNet(self.env.observation_space,
                      self.env.action_space, 32, 32)
        lagged_qf_net = QNet(self.env.observation_space,
                             self.env.action_space, 32, 32)
        lagged_qf_net.load_state_dict(qf_net.state_dict())
        targ_qf1_net = QNet(self.env.observation_space,
                            self.env.action_space, 32, 32)
        targ_qf1_net.load_state_dict(qf_net.state_dict())
        targ_qf2_net = QNet(self.env.observation_space,
                            self.env.action_space, 32, 32)
        targ_qf2_net.load_state_dict(lagged_qf_net.state_dict())
        qf = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, qf_net)
        lagged_qf = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, lagged_qf_net)
        targ_qf1 = CEMDeterministicSAVfunc(self.env.observation_space, self.env.action_space, targ_qf1_net, num_sampling=60,
                                           num_best_sampling=6, num_iter=2,
                                           multivari=False)
        targ_qf2 = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, targ_qf2_net)

        pol = ArgmaxQfPol(self.env.observation_space,
                          self.env.action_space, targ_qf1, eps=0.2)

        sampler = EpiSampler(self.env, pol, num_parallel=1)

        optim_qf = torch.optim.Adam(qf_net.parameters(), 3e-4)

        epis = sampler.sample(pol, max_steps=32)

        traj = Traj()
        traj.add_epis(epis)
        traj = ef.add_next_obs(traj)
        traj.register_epis()

        result_dict = qtopt.train(
            traj, qf, lagged_qf, targ_qf1, targ_qf2,
            optim_qf, 1000, 32,
            0.9999, 0.995, 'mse'
        )

        del sampler


class TestOnpolicyDistillation(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('Pendulum-v0')

    def test_learning(self):
        t_pol_net = PolNet(self.env.observation_space,
                           self.env.action_space, h1=200, h2=100)
        s_pol_net = PolNet(self.env.observation_space,
                           self.env.action_space, h1=190, h2=90)

        t_pol = GaussianPol(
            self.env.observation_space, self.env.action_space, t_pol_net)
        s_pol = GaussianPol(
            self.env.observation_space, self.env.action_space, s_pol_net)

        student_sampler = EpiSampler(self.env, s_pol, num_parallel=1)

        optim_pol = torch.optim.Adam(s_pol.parameters(), 3e-4)

        epis = student_sampler.sample(s_pol, max_steps=32)

        traj = Traj()
        traj.add_epis(epis)

        traj = ef.compute_h_masks(traj)
        traj.register_epis()
        result_dict = on_pol_teacher_distill.train(
            traj=traj,
            student_pol=s_pol,
            teacher_pol=t_pol,
            student_optim=optim_pol,
            epoch=1,
            batchsize=32)

        del student_sampler


class TestBehaviorClone(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('Pendulum-v0')

    def test_learning(self):
        pol_net = PolNet(self.env.observation_space,
                         self.env.action_space, h1=32, h2=32)
        pol = GaussianPol(self.env.observation_space,
                          self.env.action_space, pol_net)

        sampler = EpiSampler(self.env, pol, num_parallel=1)

        optim_pol = torch.optim.Adam(pol_net.parameters(), 3e-4)

        with open(os.path.join('data/expert_epis', 'Pendulum-v0_2epis.pkl'), 'rb') as f:
            expert_epis = pickle.load(f)
        train_epis, test_epis = ef.train_test_split(
            expert_epis, train_size=0.7)
        train_traj = Traj()
        train_traj.add_epis(train_epis)
        train_traj.register_epis()
        test_traj = Traj()
        test_traj.add_epis(test_epis)
        test_traj.register_epis()

        result_dict = behavior_clone.train(
            train_traj, pol, optim_pol,
            256
        )

        del sampler


class TestGAIL(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('Pendulum-v0')

    def test_learning(self):
        pol_net = PolNet(self.env.observation_space,
                         self.env.action_space, h1=32, h2=32)
        pol = GaussianPol(self.env.observation_space,
                          self.env.action_space, pol_net)

        vf_net = VNet(self.env.observation_space)
        vf = DeterministicSVfunc(self.env.observation_space, vf_net)

        discrim_net = DiscrimNet(
            self.env.observation_space, self.env.action_space, h1=32, h2=32)
        discrim = DeterministicSAVfunc(
            self.env.observation_space, self.env.action_space, discrim_net)

        sampler = EpiSampler(self.env, pol, num_parallel=1)

        optim_vf = torch.optim.Adam(vf_net.parameters(), 3e-4)
        optim_discrim = torch.optim.Adam(discrim_net.parameters(), 3e-4)

        with open(os.path.join('data/expert_epis', 'Pendulum-v0_2epis.pkl'), 'rb') as f:
            expert_epis = pickle.load(f)
        expert_traj = Traj()
        expert_traj.add_epis(expert_epis)
        expert_traj.register_epis()

        epis = sampler.sample(pol, max_steps=32)

        agent_traj = Traj()
        agent_traj.add_epis(epis)
        agent_traj = ef.compute_pseudo_rews(agent_traj, discrim)
        agent_traj = ef.compute_vs(agent_traj, vf)
        agent_traj = ef.compute_rets(agent_traj, 0.99)
        agent_traj = ef.compute_advs(agent_traj, 0.99, 0.95)
        agent_traj = ef.centerize_advs(agent_traj)
        agent_traj = ef.compute_h_masks(agent_traj)
        agent_traj.register_epis()

        result_dict = gail.train(agent_traj, expert_traj, pol, vf, discrim, optim_vf, optim_discrim,
                                 rl_type='trpo',
                                 epoch=1,
                                 batch_size=32, discrim_batch_size=32,
                                 discrim_step=1,
                                 pol_ent_beta=1e-3, discrim_ent_beta=1e-5)

        del sampler


class TestAIRL(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('Pendulum-v0')

    def test_learning(self):
        pol_net = PolNet(self.env.observation_space,
                         self.env.action_space, h1=32, h2=32)
        pol = GaussianPol(self.env.observation_space,
                          self.env.action_space, pol_net)

        vf_net = VNet(self.env.observation_space)
        vf = DeterministicSVfunc(self.env.observation_space, vf_net)

        rewf_net = VNet(self.env.observation_space, h1=32, h2=32)
        rewf = DeterministicSVfunc(self.env.observation_space, rewf_net)
        shaping_vf_net = VNet(self.env.observation_space, h1=32, h2=32)
        shaping_vf = DeterministicSVfunc(
            self.env.observation_space, shaping_vf_net)

        sampler = EpiSampler(self.env, pol, num_parallel=1)

        optim_vf = torch.optim.Adam(vf_net.parameters(), 3e-4)
        optim_discrim = torch.optim.Adam(
            list(rewf_net.parameters()) + list(shaping_vf_net.parameters()), 3e-4)

        with open(os.path.join('data/expert_epis', 'Pendulum-v0_2epis.pkl'), 'rb') as f:
            expert_epis = pickle.load(f)
        expert_traj = Traj()
        expert_traj.add_epis(expert_epis)
        expert_traj = ef.add_next_obs(expert_traj)
        expert_traj.register_epis()

        epis = sampler.sample(pol, max_steps=32)

        agent_traj = Traj()
        agent_traj.add_epis(epis)
        agent_traj = ef.add_next_obs(agent_traj)
        agent_traj = ef.compute_pseudo_rews(
            agent_traj, rew_giver=rewf, state_only=True)
        agent_traj = ef.compute_vs(agent_traj, vf)
        agent_traj = ef.compute_rets(agent_traj, 0.99)
        agent_traj = ef.compute_advs(agent_traj, 0.99, 0.95)
        agent_traj = ef.centerize_advs(agent_traj)
        agent_traj = ef.compute_h_masks(agent_traj)
        agent_traj.register_epis()

        result_dict = airl.train(agent_traj, expert_traj, pol, vf, optim_vf, optim_discrim,
                                 rewf=rewf, shaping_vf=shaping_vf,
                                 rl_type='trpo',
                                 epoch=1,
                                 batch_size=32, discrim_batch_size=32,
                                 discrim_step=1,
                                 pol_ent_beta=1e-3, gamma=0.99)

        del sampler


class TestMPC(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('Pendulum-v0')

    def add_noise_to_init_obs(self, epis, std):
        with torch.no_grad():
            for epi in epis:
                epi['obs'][0] += np.random.normal(0, std, epi['obs'][0].shape)
        return epis

    def test_learning(self):
        def rew_func(next_obs, acs, mean_obs=0., std_obs=1., mean_acs=0., std_acs=1.):
            next_obs = next_obs * std_obs + mean_obs
            acs = acs * std_acs + mean_acs
            # Pendulum
            rews = -(torch.acos(next_obs[:, 0].clamp(min=-1, max=1))**2 +
                     0.1 * (next_obs[:, 2].clamp(min=-8, max=8)**2) + 0.001 * acs.squeeze(-1)**2)
            rews = rews.squeeze(0)

            return rews

        # init models
        dm_net = ModelNet(self.env.observation_space, self.env.action_space)
        dm = DeterministicSModel(
            self.env.observation_space, self.env.action_space, dm_net, rnn=False)

        mpc_pol = MPCPol(self.env.observation_space, self.env.action_space,
                         dm_net, rew_func, 1, 1)
        optim_dm = torch.optim.Adam(dm_net.parameters(), 1e-3)

        # sample with mpc policy
        sampler = EpiSampler(
            self.env, mpc_pol, num_parallel=1)
        epis = sampler.sample(
            mpc_pol, max_epis=1)

        traj = Traj()
        traj.add_epis(epis)
        traj = ef.add_next_obs(traj)
        traj = ef.compute_h_masks(traj)
        traj.register_epis()

        # train
        result_dict = mpc.train_dm(
            traj, dm, optim_dm, epoch=1, batch_size=1)

        del sampler

    def test_learning_rnn(self):
        def rew_func(next_obs, acs, mean_obs=0., std_obs=1., mean_acs=0., std_acs=1.):
            next_obs = next_obs * std_obs + mean_obs
            acs = acs * std_acs + mean_acs
            # Pendulum
            rews = -(torch.acos(next_obs[:, 0].clamp(min=-1, max=1))**2 +
                     0.1 * (next_obs[:, 2].clamp(min=-8, max=8)**2) + 0.001 * acs.squeeze(-1)**2)
            rews = rews.squeeze(0)

            return rews
        # init models
        dm_net = ModelNetLSTM(self.env.observation_space,
                              self.env.action_space)
        dm = DeterministicSModel(
            self.env.observation_space, self.env.action_space, dm_net, rnn=True)

        mpc_pol = MPCPol(self.env.observation_space, self.env.action_space, dm_net, rew_func,
                         1, 1, mean_obs=0., std_obs=1., mean_acs=0., std_acs=1., rnn=True)
        optim_dm = torch.optim.Adam(dm_net.parameters(), 1e-3)

        # sample with mpc policy
        sampler = EpiSampler(
            self.env, mpc_pol, num_parallel=1)
        epis = sampler.sample(
            mpc_pol, max_epis=1)

        traj = Traj()
        traj.add_epis(epis)
        traj = ef.add_next_obs(traj)
        traj = ef.compute_h_masks(traj)
        traj.register_epis()
        traj.add_traj(traj)

        # train
        result_dict = mpc.train_dm(
            traj, dm, optim_dm, epoch=1, batch_size=1)

        del sampler


class TestR2D2SAC(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('Pendulum-v0')

    def test_learning(self):
        pol_net = PolNetLSTM(
            self.env.observation_space, self.env.action_space, h_size=32, cell_size=32)
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
                traj, qfs[i], hs_name='q_hs' + str(i), input_acs=True)
            traj = ef.compute_hs(
                traj, targ_qfs[i], hs_name='targ_q_hs' + str(i), input_acs=True)
        traj.register_epis()

        result_dict = r2d2_sac.train(
            traj,
            pol, qfs, targ_qfs, log_alpha,
            optim_pol, optim_qfs, optim_alpha,
            2, 32, 4, 2,
            0.01, 0.99, 2,
        )

        del sampler


class TestDIAYN(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('Pendulum-v0')
        self.env = SkillEnv(self.env, num_skill=4)

    def test_learning(self):
        observation_space = self.env.real_observation_space
        skill_space = self.env.skill_space
        ob_skill_space = self.env.observation_space
        action_space = self.env.action_space
        ob_dim = ob_skill_space.shape[0] - 4
        f_dim = ob_dim
        def discrim_f(x): return x

        pol_net = PolNet(ob_skill_space, action_space)
        pol = GaussianPol(ob_skill_space, action_space, pol_net)
        qf_net1 = QNet(ob_skill_space, action_space)
        qf1 = DeterministicSAVfunc(ob_skill_space, action_space, qf_net1)
        targ_qf_net1 = QNet(ob_skill_space, action_space)
        targ_qf_net1.load_state_dict(qf_net1.state_dict())
        targ_qf1 = DeterministicSAVfunc(
            ob_skill_space, action_space, targ_qf_net1)
        qf_net2 = QNet(ob_skill_space, action_space)
        qf2 = DeterministicSAVfunc(ob_skill_space, action_space, qf_net2)
        targ_qf_net2 = QNet(ob_skill_space, action_space)
        targ_qf_net2.load_state_dict(qf_net2.state_dict())
        targ_qf2 = DeterministicSAVfunc(
            ob_skill_space, action_space, targ_qf_net2)
        qfs = [qf1, qf2]
        targ_qfs = [targ_qf1, targ_qf2]
        log_alpha = nn.Parameter(torch.ones(()))

        high = np.array([np.finfo(np.float32).max] * f_dim)
        f_space = gym.spaces.Box(-high, high, dtype=np.float32)
        discrim_net = DiaynDiscrimNet(
            f_space, skill_space, h_size=100, discrim_f=discrim_f)
        discrim = DeterministicSVfunc(f_space, discrim_net)

        optim_pol = torch.optim.Adam(pol_net.parameters(), 1e-4)
        optim_qf1 = torch.optim.Adam(qf_net1.parameters(), 3e-4)
        optim_qf2 = torch.optim.Adam(qf_net2.parameters(), 3e-4)
        optim_qfs = [optim_qf1, optim_qf2]
        optim_alpha = torch.optim.Adam([log_alpha], 1e-4)
        optim_discrim = torch.optim.SGD(discrim.parameters(),
                                        lr=0.001, momentum=0.9)

        off_traj = Traj()
        sampler = EpiSampler(self.env, pol, num_parallel=1)

        epis = sampler.sample(pol, max_steps=200)
        on_traj = Traj()
        on_traj.add_epis(epis)
        on_traj = ef.add_next_obs(on_traj)
        on_traj = ef.compute_diayn_rews(
            on_traj, lambda x: diayn_sac.calc_rewards(x, 4, discrim))
        on_traj.register_epis()
        off_traj.add_traj(on_traj)
        step = on_traj.num_step
        log_alpha = nn.Parameter(np.log(0.1) * torch.ones(()))  # fix alpha
        result_dict = diayn_sac.train(
            off_traj, pol, qfs, targ_qfs, log_alpha,
            optim_pol, optim_qfs, optim_alpha,
            step, 128, 5e-3, 0.99, 1, discrim, 4, True)
        discrim_losses = diayn.train(
            discrim, optim_discrim, on_traj, 32, 100, 4)

        del sampler


if __name__ == '__main__':
    t = TestDDPG()
    t.setUp()
    t.test_learning()
    t.tearDown()

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

import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import machina as mc
from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol
from machina.algos import ppo_clip, ppo_kl
from machina.vfuncs import DeterministicSVfunc
from machina.envs import GymEnv, C2DEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina.misc import logger
from machina.utils import measure, set_device

from simple_net import PolNet, VNet, PolNetLSTM, VNetLSTM


class TestPPOContinuous(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('Pendulum-v0')

    def test_learning(self):
        pol_net = PolNet(self.env.ob_space, self.env.ac_space, h1=32, h2=32)
        pol = GaussianPol(self.env.ob_space, self.env.ac_space, pol_net)

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
        result_dict = ppo_kl.train(traj=traj, pol=pol, vf=vf, kl_beta=0.1, kl_targ=0.2,
                                    optim_pol=optim_pol, optim_vf=optim_vf, epoch=1, batch_size=32, max_grad_norm=10)

        del sampler

    def test_learning_rnn(self):
        pol_net = PolNetLSTM(self.env.ob_space, self.env.ac_space, h_size=32, cell_size=32)
        pol = GaussianPol(self.env.ob_space, self.env.ac_space, pol_net, rnn=True)

        vf_net = VNetLSTM(self.env.ob_space, h_size=32, cell_size=32)
        vf = DeterministicSVfunc(self.env.ob_space, vf_net, rnn=True)

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
        pol_net = PolNet(self.env.ob_space, self.env.ac_space, h1=32, h2=32)
        pol = CategoricalPol(self.env.ob_space, self.env.ac_space, pol_net)

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
        result_dict = ppo_kl.train(traj=traj, pol=pol, vf=vf, kl_beta=0.1, kl_targ=0.2,
                                    optim_pol=optim_pol, optim_vf=optim_vf, epoch=1, batch_size=32, max_grad_norm=10)

        del sampler

    def test_learning_rnn(self):
        pol_net = PolNetLSTM(self.env.ob_space, self.env.ac_space, h_size=32, cell_size=32)
        pol = CategoricalPol(self.env.ob_space, self.env.ac_space, pol_net, rnn=True)

        vf_net = VNetLSTM(self.env.ob_space, h_size=32, cell_size=32)
        vf = DeterministicSVfunc(self.env.ob_space, vf_net, rnn=True)

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

class TestTRPOContinuous(unittest.TestCase):
    def setUp(self):
        self.env = GymEnv('Pendulum-v0')

    def test_learning(self):
        pol_net = PolNet(self.env.ob_space, self.env.ac_space, h1=32, h2=32)
        pol = GaussianPol(self.env.ob_space, self.env.ac_space, pol_net)

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
        result_dict = ppo_kl.train(traj=traj, pol=pol, vf=vf, kl_beta=0.1, kl_targ=0.2,
                                    optim_pol=optim_pol, optim_vf=optim_vf, epoch=1, batch_size=32, max_grad_norm=10)

        del sampler

    def test_learning_rnn(self):
        pol_net = PolNetLSTM(self.env.ob_space, self.env.ac_space, h_size=32, cell_size=32)
        pol = GaussianPol(self.env.ob_space, self.env.ac_space, pol_net, rnn=True)

        vf_net = VNetLSTM(self.env.ob_space, h_size=32, cell_size=32)
        vf = DeterministicSVfunc(self.env.ob_space, vf_net, rnn=True)

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
        pol_net = PolNet(self.env.ob_space, self.env.ac_space, h1=32, h2=32)
        pol = CategoricalPol(self.env.ob_space, self.env.ac_space, pol_net)

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
        result_dict = ppo_kl.train(traj=traj, pol=pol, vf=vf, kl_beta=0.1, kl_targ=0.2,
                                    optim_pol=optim_pol, optim_vf=optim_vf, epoch=1, batch_size=32, max_grad_norm=10)

        del sampler

    # TODO: Error of github's #74
    #def test_learning_rnn(self):
    #    pol_net = PolNetLSTM(self.env.ob_space, self.env.ac_space, h_size=32, cell_size=32)
    #    pol = CategoricalPol(self.env.ob_space, self.env.ac_space, pol_net, rnn=True)

    #    vf_net = VNetLSTM(self.env.ob_space, h_size=32, cell_size=32)
    #    vf = DeterministicSVfunc(self.env.ob_space, vf_net, rnn=True)

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

if __name__ == '__main__':
    ppo_continuous = TestPPOContinuous()
    ppo_continuous.setUp()
    ppo_continuous.test_learning()
    ppo_continuous.test_learning_rnn()
    ppo_continuous.tearDown()
    ppo_discrete = TestPPODiscrete()
    ppo_discrete.setUp()
    ppo_discrete.test_learning()
    ppo_discrete.test_learning_rnn()
    ppo_discrete.tearDown()

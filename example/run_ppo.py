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

import argparse
import json
import os
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import pybullet_envs

import machina as mc
from machina.pols import GaussianPol, CategoricalPol
from machina.algos import ppo_clip, ppo_kl
from machina.prepro import BasePrePro
from machina.vfuncs import DeterministicVfunc
from machina.envs import GymEnv
from machina.data import Data, compute_vs, compute_rets, compute_advs, centerize_advs, add_h_masks
from machina.samplers import BatchSampler, ParallelSampler
from machina.misc import logger
from machina.utils import measure, set_device
from machina.nets.simple_net import PolNet, VNet, PolNetLSTM, VNetLSTM

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage')
parser.add_argument('--env_name', type=str, default='Pendulum-v0')
parser.add_argument('--roboschool', action='store_true', default=False)
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_episodes', type=int, default=1000000)
parser.add_argument('--use_parallel_sampler',
                    action='store_true', default=False)

parser.add_argument('--max_samples_per_iter', type=int, default=5000)
parser.add_argument('--max_episodes_per_iter', type=int, default=250)
parser.add_argument('--epoch_per_iter', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--pol_lr', type=float, default=1e-4)
parser.add_argument('--vf_lr', type=float, default=3e-4)
parser.add_argument('--use_prepro', action='store_true', default=False)
parser.add_argument('--cuda', type=int, default=-1)

parser.add_argument('--rnn', action='store_true', default=False)
parser.add_argument('--max_grad_norm', type=float, default=10)

parser.add_argument('--ppo_type', type=str,
                    choices=['clip', 'kl'], default='clip')

parser.add_argument('--clip_param', type=float, default=0.2)

parser.add_argument('--kl_targ', type=float, default=0.01)
parser.add_argument('--init_kl_beta', type=float, default=1)

parser.add_argument('--gamma', type=float, default=0.995)
parser.add_argument('--lam', type=float, default=1)
args = parser.parse_args()

if not os.path.exists(args.log):
    os.mkdir(args.log)

with open(os.path.join(args.log, 'args.json'), 'w') as f:
    json.dump(vars(args), f)
pprint(vars(args))

if not os.path.exists(os.path.join(args.log, 'models')):
    os.mkdir(os.path.join(args.log, 'models'))

np.random.seed(args.seed)
torch.manual_seed(args.seed)

device_name = 'cpu' if args.cuda < 0 else "cuda:{}".format(args.cuda)
device = torch.device(device_name)
set_device(device)

if args.roboschool:
    import roboschool

score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)

env = GymEnv(args.env_name, log_dir=os.path.join(
    args.log, 'movie'), record_video=args.record)
env.env.seed(args.seed)

ob_space = env.observation_space
ac_space = env.action_space

if args.rnn:
    pol_net = PolNetLSTM(ob_space, ac_space, h_size=256, cell_size=256)
else:
    pol_net = PolNet(ob_space, ac_space)
if isinstance(ac_space, gym.spaces.Box):
    pol = GaussianPol(ob_space, ac_space, pol_net, rnn=args.rnn)
else:
    pol = CategoricalPol(ob_space, ac_space, pol_net, rnn=args.rnn)

if args.rnn:
    vf_net = VNetLSTM(ob_space, h_size=256, cell_size=256)
else:
    vf_net = VNet(ob_space)
vf = DeterministicVfunc(ob_space, vf_net, args.rnn)
prepro = BasePrePro(ob_space)
if args.use_parallel_sampler:
    sampler = ParallelSampler(
        env, pol, args.max_samples_per_iter, args.max_episodes_per_iter, seed=args.seed)
else:
    sampler = BatchSampler(env)
optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)
optim_vf = torch.optim.Adam(vf_net.parameters(), args.vf_lr)

total_epi = 0
total_step = 0
max_rew = -1e6
kl_beta = args.init_kl_beta
while args.max_episodes > total_epi:
    with measure('sample'):
        if args.use_prepro:
            epis = sampler.sample(pol, args.max_samples_per_iter,
                                  args.max_episodes_per_iter, prepro.prepro_with_update)
        else:
            epis = sampler.sample(
                pol, args.max_samples_per_iter, args.max_episodes_per_iter)
    with measure('train'):
        data = Data()
        data.add_epis(epis)
        data = compute_vs(data, vf)
        data = compute_rets(data, args.gamma)
        data = compute_advs(data, args.gamma, args.lam)
        data = centerize_advs(data)
        data = add_h_masks(data)
        data.register_epis()
        if args.ppo_type == 'clip':
            result_dict = ppo_clip.train(data=data, pol=pol, vf=vf, clip_param=args.clip_param,
                                         optim_pol=optim_pol, optim_vf=optim_vf, epoch=args.epoch_per_iter, batch_size=args.batch_size, max_grad_norm=args.max_grad_norm)
        else:
            result_dict = ppo_kl.train(data=data, pol=pol, vf=vf, kl_beta=kl_beta, kl_targ=args.kl_targ,
                                       optim_pol=optim_pol, optim_vf=optim_vf, epoch=args.epoch_per_iter, batch_size=args.batch_size, max_grad_norm=args.max_grad_norm)
            kl_beta = result_dict['new_kl_beta']
    total_epi += data.num_epi
    step = sum([len(epi['rews']) for epi in epis])
    total_step += step
    rewards = [np.sum(epi['rews']) for epi in epis]
    mean_rew = np.mean(rewards)
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

    if mean_rew > max_rew:
        torch.save(pol.state_dict(), os.path.join(
            args.log, 'models', 'pol_max.pkl'))
        torch.save(vf.state_dict(), os.path.join(
            args.log, 'models', 'vf_max.pkl'))
        torch.save(optim_pol.state_dict(), os.path.join(
            args.log, 'models', 'optim_pol_max.pkl'))
        torch.save(optim_vf.state_dict(), os.path.join(
            args.log, 'models', 'optim_vf_max.pkl'))
        max_rew = mean_rew

    torch.save(pol.state_dict(), os.path.join(
        args.log, 'models', 'pol_last.pkl'))
    torch.save(vf.state_dict(), os.path.join(
        args.log, 'models', 'vf_last.pkl'))
    torch.save(optim_pol.state_dict(), os.path.join(
        args.log, 'models', 'optim_pol_last.pkl'))
    torch.save(optim_vf.state_dict(), os.path.join(
        args.log, 'models', 'optim_vf_last.pkl'))
    del data
del sampler

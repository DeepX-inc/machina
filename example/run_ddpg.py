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
import copy
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import pybullet_envs

import machina as mc
from machina.pols import DeterministicPol, OrnsteinUhlenbeckActionNoise
from machina.algos import ddpg
from machina.prepro import BasePrePro
from machina.qfuncs import DeterministicQfunc
from machina.envs import GymEnv
from machina.data import ReplayData, GAEData
from machina.samplers import BatchSampler, ParallelSampler
from machina.misc import logger
from machina.utils import set_device, measure
from machina.nets import DeterministicPolNet, QNet

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage')
parser.add_argument('--env_name', type=str, default='Pendulum-v0')
parser.add_argument('--roboschool', action='store_true', default=False)
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--episode', type=int, default=1000000)
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_episodes', type=int, default=1000000)
parser.add_argument('--use_parallel_sampler', action='store_true', default=False)

parser.add_argument('--max_data_size', type=int, default=1000000)
parser.add_argument('--min_data_size', type=int, default=10000)
parser.add_argument('--max_samples_per_iter', type=int, default=2000)
parser.add_argument('--max_episodes_per_iter', type=int, default=10000)
parser.add_argument('--epoch_per_iter', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--pol_lr', type=float, default=1e-4)
parser.add_argument('--qf_lr', type=float, default=1e-3)
parser.add_argument('--use_prepro', action='store_true', default=False)
parser.add_argument('--cuda', type=int, default=-1)
parser.add_argument('--h1', type=int, default=32)
parser.add_argument('--h2', type=int, default=32)

parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lam', type=float, default=1)
args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available() else -1
set_device(args.cuda)

if not os.path.exists(args.log):
    os.mkdir(args.log)

with open(os.path.join(args.log, 'args.json'), 'w') as f:
    json.dump(vars(args), f)
pprint(vars(args))

if not os.path.exists(os.path.join(args.log, 'models')):
    os.mkdir(os.path.join(args.log, 'models'))

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.roboschool:
    import roboschool

score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)

env = GymEnv(args.env_name, log_dir=os.path.join(args.log, 'movie'), record_video=args.record)
env.env.seed(args.seed)

ob_space = env.observation_space
ac_space = env.action_space

pol_net = DeterministicPolNet(ob_space, ac_space, args.h1, args.h2)
noise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(ac_space.shape[0]))
pol = DeterministicPol(ob_space, ac_space, pol_net, noise)
targ_pol = copy.deepcopy(pol)
qf_net = QNet(ob_space, ac_space, args.h1, args.h2)
qf = DeterministicQfunc(ob_space, ac_space, qf_net)
targ_qf = copy.deepcopy(qf)
prepro = BasePrePro(ob_space)
if args.use_parallel_sampler:
    sampler = ParallelSampler(env)
else:
    sampler = BatchSampler(env)
optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)
optim_qf = torch.optim.Adam(qf_net.parameters(), args.qf_lr)
off_data = ReplayData(max_data_size=args.max_data_size + 1, ob_dim=ob_space.shape[0], ac_dim=ac_space.shape[0])

total_epi = 0
total_step = 0
max_rew = -1e6

count = 0

while args.max_episodes > total_epi:
    with measure('sample'):
        if args.use_prepro:
            paths = sampler.sample(pol, args.max_samples_per_iter, args.max_episodes_per_iter, prepro.prepro_with_update)
        else:
            paths = sampler.sample(pol, args.max_samples_per_iter, args.max_episodes_per_iter)
    off_data.add_paths(paths)

    epi = len(paths)
    total_epi += epi
    step = sum([len(path['rews']) for path in paths])
    total_step += step

    with measure('train'):
        result_dict = ddpg.train(
            off_data,
            pol, targ_pol, qf, targ_qf,
            optim_pol, optim_qf, step, args.batch_size,
            args.tau, args.gamma, args.lam
        )

    rewards = [np.sum(path['rews']) for path in paths]
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

    mean_rew = np.mean([np.sum(path['rews']) for path in paths])
    if mean_rew > max_rew:
        torch.save(pol.state_dict(), os.path.join(args.log, 'models', 'pol_max.pkl'))
        torch.save(qf.state_dict(), os.path.join(args.log, 'models',  'qf_max.pkl'))
        torch.save(optim_pol.state_dict(), os.path.join(args.log, 'models',  'optim_pol_max.pkl'))
        torch.save(optim_qf.state_dict(), os.path.join(args.log, 'models',  'optim_qf_max.pkl'))
        max_rew = mean_rew

    torch.save(pol.state_dict(), os.path.join(args.log, 'models',  'pol_last.pkl'))
    torch.save(qf.state_dict(), os.path.join(args.log, 'models', 'qf_last.pkl'))
    torch.save(optim_pol.state_dict(), os.path.join(args.log, 'models',  'optim_pol_last.pkl'))
    torch.save(optim_qf.state_dict(), os.path.join(args.log, 'models',  'optim_qf_last.pkl'))

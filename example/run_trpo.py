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
from machina.algos import trpo
from machina.prepro import BasePrePro
from machina.vfuncs import DeterministicVfunc
from machina.envs import GymEnv
from machina.data import Data, compute_vs, compute_rets, compute_advs, centerize_advs, add_h_masks
from machina.samplers import EpiSampler
from machina.misc import logger
from machina.utils import measure
from machina.nets import PolNet, VNet, PolNetLSTM, VNetLSTM

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage')
parser.add_argument('--env_name', type=str, default='Pendulum-v0')
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--episode', type=int, default=1000000)
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_episodes', type=int, default=1000000)
parser.add_argument('--num_parallel', type=int, default=4)

parser.add_argument('--max_episodes_per_iter', type=int, default=256)
parser.add_argument('--epoch_per_iter', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--vf_lr', type=float, default=3e-4)
parser.add_argument('--use_prepro', action='store_true', default=False)
parser.add_argument('--rnn', action='store_true', default=False)

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

score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)

env = GymEnv(args.env_name, log_dir=os.path.join(args.log, 'movie'), record_video=args.record)
env.env.seed(args.seed)

ob_space = env.observation_space
ac_space = env.action_space

if args.rnn:
    pol_net = PolNetLSTM(ob_space, ac_space, h_size=256, cell_size=256)
else:
    pol_net = PolNet(ob_space, ac_space)
if isinstance(ac_space, gym.spaces.Box):
    pol = GaussianPol(ob_space, ac_space, pol_net, args.rnn)
else:
    pol = CategoricalPol(ob_space, ac_space, pol_net, args.rnn)

if args.rnn:
    vf_net = VNetLSTM(ob_space, h_size=256, cell_size=256)
else:
    vf_net = VNet(ob_space)
vf = DeterministicVfunc(ob_space, vf_net, args.rnn)

prepro = BasePrePro(ob_space)

sampler = EpiSampler(env, pol, num_parallel=args.num_parallel, prepro=prepro, seed=args.seed)
optim_vf = torch.optim.Adam(vf_net.parameters(), args.vf_lr)

total_epi = 0
total_step = 0
max_rew = -1e6
while args.max_episodes > total_epi:
    with measure('sample'):
        if args.use_prepro:
            epis = sampler.sample(pol, args.max_episodes_per_iter, prepro.prepro_with_update)
        else:
            epis = sampler.sample(pol, args.max_episodes_per_iter)
    with measure('train'):
        data = Data()
        data.add_epis(epis)
        data = compute_vs(data, vf)
        data = compute_rets(data, args.gamma)
        data = compute_advs(data, args.gamma, args.lam)
        data = centerize_advs(data)
        data = add_h_masks(data)
        data.register_epis()
        result_dict = trpo.train(data, pol, vf, optim_vf, args.epoch_per_iter, args.batch_size)

    total_epi += data.num_epi
    step = sum([len(epi['rews']) for epi in epis])
    total_step += step
    rewards = [np.sum(epi['rews']) for epi in epis]
    mean_rew = np.mean(rewards)
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

    mean_rew = np.mean([np.sum(epi['rews']) for epi in epis])
    if mean_rew > max_rew:
        torch.save(pol.state_dict(), os.path.join(args.log, 'models', 'pol_max.pkl'))
        torch.save(vf.state_dict(), os.path.join(args.log, 'models', 'vf_max.pkl'))
        torch.save(optim_vf.state_dict(), os.path.join(args.log, 'models', 'optim_vf_max.pkl'))
        max_rew = mean_rew

    torch.save(pol.state_dict(), os.path.join(args.log, 'models', 'pol_last.pkl'))
    torch.save(vf.state_dict(), os.path.join(args.log, 'models', 'vf_last.pkl'))
    torch.save(optim_vf.state_dict(), os.path.join(args.log, 'models', 'optim_vf_last.pkl'))
    del data


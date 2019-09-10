"""
An example of Trust Region Policy Optimization.
"""

import argparse
import json
import os
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import machina as mc
from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol
from machina.algos import trpo
from machina.vfuncs import DeterministicSVfunc
from machina.envs import GymEnv, C2DEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import measure

from simple_net import PolNet, VNet, PolNetLSTM, VNetLSTM

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage')
parser.add_argument('--env_name', type=str, default='Pendulum-v0')
parser.add_argument('--c2d', action='store_true', default=False)
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_epis', type=int, default=1000000)
parser.add_argument('--num_parallel', type=int, default=4)

parser.add_argument('--max_steps_per_iter', type=int, default=10000)
parser.add_argument('--epoch_per_iter', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--vf_lr', type=float, default=3e-4)
parser.add_argument('--rnn', action='store_true', default=False)
parser.add_argument('--rnn_batch_size', type=int, default=8,
                    help='Number of sequences included in batch of rnn.')

parser.add_argument('--gamma', type=float, default=0.995)
parser.add_argument('--lam', type=float, default=1)
args = parser.parse_args()

if not os.path.exists(args.log):
    os.makedirs(args.log)

with open(os.path.join(args.log, 'args.json'), 'w') as f:
    json.dump(vars(args), f)
pprint(vars(args))

if not os.path.exists(os.path.join(args.log, 'models')):
    os.makedirs(os.path.join(args.log, 'models'))

np.random.seed(args.seed)
torch.manual_seed(args.seed)

score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)
logger.add_tensorboard_output(args.log)

env = GymEnv(args.env_name, log_dir=os.path.join(
    args.log, 'movie'), record_video=args.record)
env.env.seed(args.seed)
if args.c2d:
    env = C2DEnv(env)

observation_space = env.observation_space
action_space = env.action_space

if args.rnn:
    pol_net = PolNetLSTM(observation_space, action_space,
                         h_size=256, cell_size=256)
else:
    pol_net = PolNet(observation_space, action_space)
if isinstance(action_space, gym.spaces.Box):
    pol = GaussianPol(observation_space, action_space, pol_net, args.rnn)
elif isinstance(action_space, gym.spaces.Discrete):
    pol = CategoricalPol(observation_space, action_space, pol_net, args.rnn)
elif isinstance(action_space, gym.spaces.MultiDiscrete):
    pol = MultiCategoricalPol(
        observation_space, action_space, pol_net, args.rnn)
else:
    raise ValueError('Only Box, Discrete, and MultiDiscrete are supported')

if args.rnn:
    vf_net = VNetLSTM(observation_space, h_size=256, cell_size=256)
else:
    vf_net = VNet(observation_space)
vf = DeterministicSVfunc(observation_space, vf_net, args.rnn)

sampler = EpiSampler(env, pol, num_parallel=args.num_parallel, seed=args.seed)
optim_vf = torch.optim.Adam(vf_net.parameters(), args.vf_lr)

total_epi = 0
total_step = 0
max_rew = -1e6
while args.max_epis > total_epi:
    with measure('sample'):
        epis = sampler.sample(pol, max_steps=args.max_steps_per_iter)
    with measure('train'):
        traj = Traj()
        traj.add_epis(epis)

        traj = ef.compute_vs(traj, vf)
        traj = ef.compute_rets(traj, args.gamma)
        traj = ef.compute_advs(traj, args.gamma, args.lam)
        traj = ef.centerize_advs(traj)
        traj = ef.compute_h_masks(traj)
        traj.register_epis()

        result_dict = trpo.train(
            traj, pol, vf, optim_vf, args.epoch_per_iter, batch_size=args.batch_size if not args.rnn else args.rnn_batch_size)

    total_epi += traj.num_epi
    step = traj.num_step
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
        torch.save(optim_vf.state_dict(), os.path.join(
            args.log, 'models', 'optim_vf_max.pkl'))
        max_rew = mean_rew

    torch.save(pol.state_dict(), os.path.join(
        args.log, 'models', 'pol_last.pkl'))
    torch.save(vf.state_dict(), os.path.join(
        args.log, 'models', 'vf_last.pkl'))
    torch.save(optim_vf.state_dict(), os.path.join(
        args.log, 'models', 'optim_vf_last.pkl'))
    del traj
del sampler

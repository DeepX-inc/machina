"""
Script for making file of expert epis.
"""
import argparse
import json
import os
from pprint import pprint
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import machina as mc
from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol, DeterministicActionNoisePol
from machina.noise import OUActionNoise
from machina.envs import GymEnv, C2DEnv
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import measure, set_device


from simple_net import PolNet, VNet, PolNetLSTM, VNetLSTM

parser = argparse.ArgumentParser()
parser.add_argument('--pol_dir', type=str, default='../data/expert_pols',
                    help='Directory path storing file of expert policy model.')
parser.add_argument('--pol_fname', type=str, default='pol_max.pkl',
                    help='File name of expert policy model.')
parser.add_argument('--epis_dir', type=str, default='../data/expert_epis',
                    help='Directory path to store file of expert trajectory.')
parser.add_argument('--epis_fname', type=str, default='',
                    help='File name of expert trajectory.')
parser.add_argument('--env_name', type=str,
                    default='Pendulum-v0', help='Name of environment.')
parser.add_argument('--c2d', action='store_true',
                    default=False, help='If True, action is discretized.')
parser.add_argument('--record', action='store_true',
                    default=False, help='If True, movie is saved.')
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_epis', type=int,
                    default=100000000, help='Number of episodes to run.')
parser.add_argument('--num_parallel', type=int, default=1,
                    help='Number of processes to sample.')
parser.add_argument('--cuda', type=int, default=-1, help='cuda device number.')
parser.add_argument('--rnn', action='store_true',
                    default=False, help='If True, network is reccurent.')
parser.add_argument('--pol_h1', type=int, default=100,
                    help='Hidden size of layer1 of policy.')
parser.add_argument('--pol_h2', type=int, default=100,
                    help='Hidden size of layer2 of policy.')

parser.add_argument('--num_epis', type=int, default=100,
                    help='Number of episodes of expert trajectories.')
parser.add_argument('--ddpg', action='store_true',
                    default=False, help='If True, policy for DDPG is used.')
args = parser.parse_args()

if not os.path.exists(args.pol_dir):
    os.mkdir(args.pol_dir)

with open(os.path.join(args.pol_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f)
pprint(vars(args))

if not os.path.exists(os.path.join(args.epis_dir)):
    os.mkdir(args.epis_dir)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

device_name = 'cpu' if args.cuda < 0 else "cuda:{}".format(args.cuda)
device = torch.device(device_name)
set_device(device)

env = GymEnv(args.env_name, log_dir=os.path.join(
    args.pol_dir, 'movie'), record_video=args.record)
env.env.seed(args.seed)
if args.c2d:
    env = C2DEnv(env)

observation_space = env.observation_space
action_space = env.action_space

if args.ddpg:
    pol_net = PolNet(observation_space, action_space, args.pol_h1,
                     args.pol_h2, deterministic=True)
    noise = OUActionNoise(action_space.shape)
    pol = DeterministicActionNoisePol(
        observation_space, action_space, pol_net, noise)
else:
    if args.rnn:
        pol_net = PolNetLSTM(observation_space, action_space,
                             h_size=256, cell_size=256)
    else:
        pol_net = PolNet(observation_space, action_space)
    if isinstance(action_space, gym.spaces.Box):
        pol = GaussianPol(observation_space, action_space, pol_net, args.rnn)
    elif isinstance(action_space, gym.spaces.Discrete):
        pol = CategoricalPol(
            observation_space, action_space, pol_net, args.rnn)
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        pol = MultiCategoricalPol(
            observation_space, action_space, pol_net, args.rnn)
    else:
        raise ValueError('Only Box, Discrete, and MultiDiscrete are supported')


sampler = EpiSampler(env, pol, num_parallel=args.num_parallel,  seed=args.seed)

with open(os.path.join(args.pol_dir, args.pol_fname), 'rb') as f:
    pol.load_state_dict(torch.load(
        f, map_location=lambda storage, location: storage))


epis = sampler.sample(pol, max_epis=args.num_epis)

filename = args.epis_fname if len(
    args.epis_fname) != 0 else env.env.spec.id + '_{}epis.pkl'.format(len(epis))
with open(os.path.join(args.epis_dir, filename), 'wb') as f:
    pickle.dump(epis, f)
rewards = [np.sum(epi['rews']) for epi in epis]
mean_rew = np.mean(rewards)
logger.log('expert_score={}'.format(mean_rew))
del sampler

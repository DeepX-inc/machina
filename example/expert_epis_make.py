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
parser.add_argument('--pol_dir', type=str, default='../data/expert_pols')
parser.add_argument('--pol_fname', type=str, default='pol_max.pkl')
parser.add_argument('--epis_dir', type=str, default='../data/expert_epis')
parser.add_argument('--epis_fname', type=str, default='')
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--num_parallel', type=int, default=8)
parser.add_argument('--h1', type=int, default=32)
parser.add_argument('--h2', type=int, default=32)
parser.add_argument('--num_epis', type=int, default=100)
parser.add_argument('--env_name', type=str, default='Pendulum-v0_100epis.pkl')
parser.add_argument('--seed', type=int, default='256')
parser.add_argument('--cuda', type=int, default='-1')
parser.add_argument('--c2d', action='store_true', default=False)
parser.add_argument('--rnn', action='store_true', default=False)
parser.add_argument('--ddpg', action='store_true', default=False)
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

ob_space = env.observation_space
ac_space = env.action_space

if args.ddpg:
    pol_net = PolNet(ob_space, ac_space, args.h1, args.h2, deterministic=True)
    noise = OUActionNoise(ac_space.shape)
    pol = DeterministicActionNoisePol(ob_space, ac_space, pol_net, noise)
else:
    if args.rnn:
        pol_net = PolNetLSTM(ob_space, ac_space, h_size=256, cell_size=256)
    else:
        pol_net = PolNet(ob_space, ac_space)
    if isinstance(ac_space, gym.spaces.Box):
        pol = GaussianPol(ob_space, ac_space, pol_net, args.rnn)
    elif isinstance(ac_space, gym.spaces.Discrete):
        pol = CategoricalPol(ob_space, ac_space, pol_net, args.rnn)
    elif isinstance(ac_space, gym.spaces.MultiDiscrete):
        pol = MultiCategoricalPol(ob_space, ac_space, pol_net, args.rnn)
    else:
        raise ValueError('Only Box, Discrete, and MultiDiscrete are supported')


sampler = EpiSampler(env, pol, num_parallel=args.num_parallel,  seed=args.seed)

with open(os.path.join(args.pol_dir, args.pol_fname), 'rb') as f:
    pol.load_state_dict(torch.load(
        f, map_location=lambda storage, location: storage))


epis = sampler.sample(pol, max_episodes=args.num_epis)

filename = args.epis_fname if len(
    args.epis_fname) != 0 else env.env.spec.id + '_{}epis.pkl'.format(len(epis))
with open(os.path.join(args.epis_dir, filename), 'wb') as f:
    pickle.dump(epis, f)
rewards = [np.sum(epi['rews']) for epi in epis]
mean_rew = np.mean(rewards)
logger.log('expert_score={}'.format(mean_rew))
del sampler

"""
An example of Behavioral Cloning.
"""

import argparse
import json
import os
import copy
from pprint import pprint
import pickle

import numpy as np
import torch
import gym

from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol, DeterministicActionNoisePol
from machina.algos import behavior_clone
from machina.envs import GymEnv, C2DEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import measure, set_device

from simple_net import PolNet, PolNetLSTM, VNet, DiscrimNet

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage',
                    help='Directory name of log.')
parser.add_argument('--env_name', type=str,
                    default='Pendulum-v0', help='Name of environment.')
parser.add_argument('--c2d', action='store_true',
                    default=False, help='If True, action is discretized.')
parser.add_argument('--record', action='store_true',
                    default=False, help='If True, movie is saved.')
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_epis', type=int,
                    default=100000000, help='Number of episodes to run.')
parser.add_argument('--num_parallel', type=int, default=4,
                    help='Number of processes to sample.')
parser.add_argument('--cuda', type=int, default=-1, help='cuda device number.')

parser.add_argument('--expert_dir', type=str, default='../data/expert_epis')
parser.add_argument('--expert_fname', type=str,
                    default='Pendulum-v0_100epis.pkl')

parser.add_argument('--max_epis_per_iter', type=int,
                    default=10, help='Number of episodes in an iteration.')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--pol_lr', type=float, default=1e-4,
                    help='Policy learning rate.')
parser.add_argument('--h1', type=int, default=32)
parser.add_argument('--h2', type=int, default=32)

parser.add_argument('--tau', type=float, default=0.001,
                    help='Coefficient of target function.')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='Discount factor.')
parser.add_argument('--lam', type=float, default=1,
                    help='Tradeoff value of bias variance.')

parser.add_argument('--train_size', type=int, default=0.7,
                    help='Size of training data.')
parser.add_argument('--check_rate', type=int, default=0.05,
                    help='Rate of performance check per epoch.')
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--deterministic', action='store_true',
                    default=False, help='If True, policy is deterministic.')
args = parser.parse_args()

device_name = 'cpu' if args.cuda < 0 else "cuda:{}".format(args.cuda)
device = torch.device(device_name)
set_device(device)

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

pol_net = PolNet(observation_space, action_space)
if isinstance(action_space, gym.spaces.Box):
    pol = GaussianPol(observation_space, action_space, pol_net)
elif isinstance(action_space, gym.spaces.Discrete):
    pol = CategoricalPol(observation_space, action_space, pol_net)
elif isinstance(action_space, gym.spaces.MultiDiscrete):
    pol = MultiCategoricalPol(observation_space, action_space, pol_net)
else:
    raise ValueError('Only Box, Discrete, and MultiDiscrete are supported')

sampler = EpiSampler(env, pol, num_parallel=args.num_parallel, seed=args.seed)
optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)

with open(os.path.join(args.expert_dir, args.expert_fname), 'rb') as f:
    expert_epis = pickle.load(f)
train_epis, test_epis = ef.train_test_split(
    expert_epis, train_size=args.train_size)
train_traj = Traj()
train_traj.add_epis(train_epis)
train_traj.register_epis()
test_traj = Traj()
test_traj.add_epis(test_epis)
test_traj.register_epis()
expert_rewards = [np.sum(epi['rews']) for epi in expert_epis]
expert_mean_rew = np.mean(expert_rewards)
logger.log('expert_score={}'.format(expert_mean_rew))
logger.log('num_train_epi={}'.format(train_traj.num_epi))

max_rew = -1e6

for curr_epoch in range(args.epoch):
    result_dict = behavior_clone.train(
        train_traj, pol, optim_pol,
        args.batch_size
    )
    test_result_dict = behavior_clone.test(test_traj, pol)

    for key in test_result_dict.keys():
        result_dict[key] = test_result_dict[key]

        if curr_epoch % int(args.check_rate * args.epoch) == 0 or curr_epoch == 0:
            with measure('sample'):
                paths = sampler.sample(
                    pol, max_epis=args.max_epis_per_iter)
            rewards = [np.sum(path['rews']) for path in paths]
            mean_rew = np.mean([np.sum(path['rews']) for path in paths])
            logger.record_results_bc(args.log, result_dict, score_file,
                                     curr_epoch, rewards,
                                     plot_title=args.env_name)

        if mean_rew > max_rew:
            torch.save(pol.state_dict(), os.path.join(
                args.log, 'models', 'pol_max.pkl'))
            torch.save(optim_pol.state_dict(), os.path.join(
                args.log, 'models', 'optim_pol_max.pkl'))
            max_rew = mean_rew

        torch.save(pol.state_dict(), os.path.join(
            args.log, 'models', 'pol_last.pkl'))
        torch.save(optim_pol.state_dict(), os.path.join(
            args.log, 'models', 'optim_pol_last.pkl'))

del sampler

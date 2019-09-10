"""
An example of Soft Actor Critic.
"""

import argparse
import copy
import json
import os
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import machina as mc
from machina.pols import GaussianPol
from machina.algos import sac
from machina.vfuncs import DeterministicSAVfunc
from machina.envs import GymEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import set_device, measure

from simple_net import PolNet, QNet, VNet


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
                    default=1000000, help='Number of episodes to run.')
parser.add_argument('--max_steps_off', type=int,
                    default=1000000000000, help='Number of episodes stored in off traj.')
parser.add_argument('--num_parallel', type=int, default=4,
                    help='Number of processes to sample.')
parser.add_argument('--cuda', type=int, default=-1, help='cuda device number.')

parser.add_argument('--max_steps_per_iter', type=int, default=10000,
                    help='Number of steps to use in an iteration.')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--sampling', type=int, default=1,
                    help='Number of sampling in calculation of expectation.')
parser.add_argument('--no_reparam', action='store_true', default=False)
parser.add_argument('--pol_lr', type=float, default=1e-4,
                    help='Policy learning rate')
parser.add_argument('--qf_lr', type=float, default=3e-4,
                    help='Q function learning rate')

parser.add_argument('--ent_alpha', type=float, default=1,
                    help='Entropy coefficient.')
parser.add_argument('--tau', type=float, default=5e-3,
                    help='Coefficient of target function.')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='Discount factor.')
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

device_name = 'cpu' if args.cuda < 0 else "cuda:{}".format(args.cuda)
device = torch.device(device_name)
set_device(device)

score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)
logger.add_tensorboard_output(args.log)

env = GymEnv(args.env_name, log_dir=os.path.join(
    args.log, 'movie'), record_video=args.record)
env.env.seed(args.seed)

observation_space = env.observation_space
action_space = env.action_space

pol_net = PolNet(observation_space, action_space)
pol = GaussianPol(observation_space, action_space, pol_net)

qf_net1 = QNet(observation_space, action_space)
qf1 = DeterministicSAVfunc(observation_space, action_space, qf_net1)
targ_qf_net1 = QNet(observation_space, action_space)
targ_qf_net1.load_state_dict(qf_net1.state_dict())
targ_qf1 = DeterministicSAVfunc(observation_space, action_space, targ_qf_net1)

qf_net2 = QNet(observation_space, action_space)
qf2 = DeterministicSAVfunc(observation_space, action_space, qf_net2)
targ_qf_net2 = QNet(observation_space, action_space)
targ_qf_net2.load_state_dict(qf_net2.state_dict())
targ_qf2 = DeterministicSAVfunc(observation_space, action_space, targ_qf_net2)

qfs = [qf1, qf2]
targ_qfs = [targ_qf1, targ_qf2]

log_alpha = nn.Parameter(torch.zeros((), device=device))

sampler = EpiSampler(env, pol, args.num_parallel, seed=args.seed)

optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)
optim_qf1 = torch.optim.Adam(qf_net1.parameters(), args.qf_lr)
optim_qf2 = torch.optim.Adam(qf_net2.parameters(), args.qf_lr)
optim_qfs = [optim_qf1, optim_qf2]
optim_alpha = torch.optim.Adam([log_alpha], args.pol_lr)

off_traj = Traj(args.max_steps_off, traj_device='cpu')

total_epi = 0
total_step = 0
max_rew = -1e6

while args.max_epis > total_epi:
    with measure('sample'):
        epis = sampler.sample(pol, max_steps=args.max_steps_per_iter)

    with measure('train'):
        on_traj = Traj(traj_device='cpu')
        on_traj.add_epis(epis)

        on_traj = ef.add_next_obs(on_traj)
        on_traj.register_epis()

        off_traj.add_traj(on_traj)

        total_epi += on_traj.num_epi
        step = on_traj.num_step
        total_step += step

        result_dict = sac.train(
            off_traj,
            pol, qfs, targ_qfs, log_alpha,
            optim_pol, optim_qfs, optim_alpha,
            step, args.batch_size,
            args.tau, args.gamma, args.sampling, not args.no_reparam
        )

    rewards = [np.sum(epi['rews']) for epi in epis]
    mean_rew = np.mean(rewards)
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

    if mean_rew > max_rew:
        torch.save(pol.state_dict(), os.path.join(
            args.log, 'models', 'pol_max.pkl'))
        torch.save(qf1.state_dict(), os.path.join(
            args.log, 'models', 'qf1_max.pkl'))
        torch.save(qf2.state_dict(), os.path.join(
            args.log, 'models', 'qf2_max.pkl'))
        torch.save(optim_pol.state_dict(), os.path.join(
            args.log, 'models', 'optim_pol_max.pkl'))
        torch.save(optim_qf1.state_dict(), os.path.join(
            args.log, 'models', 'optim_qf1_max.pkl'))
        torch.save(optim_qf2.state_dict(), os.path.join(
            args.log, 'models', 'optim_qf2_max.pkl'))
        max_rew = mean_rew

    torch.save(pol.state_dict(), os.path.join(
        args.log, 'models', 'pol_last.pkl'))
    torch.save(qf1.state_dict(), os.path.join(
        args.log, 'models', 'qf1_last.pkl'))
    torch.save(qf2.state_dict(), os.path.join(
        args.log, 'models', 'qf2_last.pkl'))
    torch.save(optim_pol.state_dict(), os.path.join(
        args.log, 'models', 'optim_pol_last.pkl'))
    torch.save(optim_qf1.state_dict(), os.path.join(
        args.log, 'models', 'optim_qf1_last.pkl'))
    torch.save(optim_qf2.state_dict(), os.path.join(
        args.log, 'models', 'optim_qf2_last.pkl'))
    del on_traj
del sampler

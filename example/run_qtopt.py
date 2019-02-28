"""
An example of QT-Opt.
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
from machina.pols import ContinuousQfPol
from machina.noise import OUActionNoise
from machina.algos import qtopt
from machina.vfuncs import DeterministicSAVfunc, CEMDeterministicSAVfunc
from machina.envs import GymEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import set_device, measure

from simple_net import QNet

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage')
parser.add_argument('--env_name', type=str, default='Pendulum-v0')
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_episodes', type=int, default=1000000)
parser.add_argument('--num_parallel', type=int, default=4)
parser.add_argument('--cuda', type=int, default=-1)

parser.add_argument('--max_steps_per_iter', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--pol_lr', type=float, default=1e-4)
parser.add_argument('--qf_lr', type=float, default=1e-3)
parser.add_argument('--h1', type=int, default=32)
parser.add_argument('--h2', type=int, default=32)

parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lag', type=int, default=6000)
parser.add_argument('--num_iter', type=int, default=2)
parser.add_argument('--num_sampling', type=int, default=60)
parser.add_argument('--num_best_sampling', type=int, default=6)
parser.add_argument('--loss_type', type=str,
                    choices=['mse', 'bce'], default='mse')
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

score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)

env = GymEnv(args.env_name, log_dir=os.path.join(
    args.log, 'movie'), record_video=args.record)
env.env.seed(args.seed)

ob_space = env.observation_space
ac_space = env.action_space

qf_net = QNet(ob_space, ac_space, args.h1, args.h2)
lagged_qf_net = QNet(ob_space, ac_space, args.h1, args.h2)
lagged_qf_net.load_state_dict(qf_net.state_dict())
targ_qf1_net = QNet(ob_space, ac_space, args.h1, args.h2)
targ_qf1_net.load_state_dict(qf_net.state_dict())
targ_qf2_net = QNet(ob_space, ac_space, args.h1, args.h2)
targ_qf2_net.load_state_dict(qf_net.state_dict())
qf = DeterministicSAVfunc(ob_space, ac_space, qf_net)
lagged_qf = DeterministicSAVfunc(ob_space, ac_space, lagged_qf_net)
targ_qf1 = CEMDeterministicSAVfunc(ob_space, ac_space, targ_qf1_net, num_sampling=args.num_sampling,
                                   num_best_sampling=args.num_best_sampling, num_iter=args.num_iter)
targ_qf2 = DeterministicSAVfunc(ob_space, ac_space, targ_qf2_net)

pol = ContinuousQfPol(ob_space, ac_space, targ_qf1)

sampler = EpiSampler(env, pol, num_parallel=args.num_parallel, seed=args.seed)

optim_qf = torch.optim.Adam(qf_net.parameters(), args.qf_lr)

off_traj = Traj()

total_epi = 0
total_step = 0
total_grad_step = 0
num_update_lagged = 0
max_rew = -1e6

while args.max_episodes > total_epi:
    with measure('sample'):
        epis = sampler.sample(pol, max_steps=args.max_steps_per_iter)
    with measure('train'):
        on_traj = Traj()
        on_traj.add_epis(epis)

        on_traj = ef.add_next_obs(on_traj)
        on_traj.register_epis()

        off_traj.add_traj(on_traj)

        total_epi += on_traj.num_epi
        step = on_traj.num_step
        total_step += step

        result_dict = qtopt.train(
            off_traj, qf, lagged_qf, targ_qf1, targ_qf2,
            optim_qf, step, args.batch_size,
            args.tau, args.gamma, loss_type=args.loss_type
        )

    total_grad_step += result_dict['grad_step']
    if total_grad_step >= args.lag * num_update_lagged:
        logger.log('Updated lagged qf!!')
        lagged_qf_net.load_state_dict(qf_net.state_dict())
        num_update_lagged += 1

    rewards = [np.sum(epi['rews']) for epi in epis]
    mean_rew = np.mean(rewards)
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

    if mean_rew > max_rew:
        torch.save(pol.state_dict(), os.path.join(
            args.log, 'models', 'pol_max.pkl'))
        torch.save(qf.state_dict(), os.path.join(
            args.log, 'models',  'qf_max.pkl'))
        torch.save(targ_qf1.state_dict(), os.path.join(
            args.log, 'models',  'targ_qf1_max.pkl'))
        torch.save(targ_qf2.state_dict(), os.path.join(
            args.log, 'models',  'targ_qf2_max.pkl'))
        torch.save(optim_qf.state_dict(), os.path.join(
            args.log, 'models',  'optim_qf_max.pkl'))
        max_rew = mean_rew

    torch.save(pol.state_dict(), os.path.join(
        args.log, 'models',  'pol_last.pkl'))
    torch.save(qf.state_dict(), os.path.join(
        args.log, 'models', 'qf_last.pkl'))
    torch.save(targ_qf1.state_dict(), os.path.join(
        args.log, 'models', 'targ_qf1_last.pkl'))
    torch.save(targ_qf2.state_dict(), os.path.join(
        args.log, 'models', 'targ_qf2_last.pkl'))
    torch.save(optim_qf.state_dict(), os.path.join(
        args.log, 'models',  'optim_qf_last.pkl'))
    del on_traj
del sampler

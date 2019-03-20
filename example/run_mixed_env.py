"""
An example of mixed environment with ppo.
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
import pybullet_envs

import machina as mc
from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol
from machina.algos import ppo_clip
from machina.vfuncs import DeterministicSVfunc
from machina.envs import GymEnv, C2DEnv, AcInObEnv, RewInObEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import measure, set_device

from simple_net import PolNet, VNet, PolNetLSTM, VNetLSTM

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage',
                    help='Directory name of log.')
parser.add_argument('--env_name', type=str,
                    default='Pendulum-v0', help='Name of environment.')
parser.add_argument('--record', action='store_true',
                    default=False, help='If True, movie is saved.')
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_epis', type=int,
                    default=1000000, help='Number of episodes to run.')
parser.add_argument('--num_parallel', type=int, default=4,
                    help='Number of processes to sample.')
parser.add_argument('--cuda', type=int, default=-1, help='cuda device number.')
parser.add_argument('--data_parallel', action='store_true', default=False,
                    help='If True, inference is done in parallel on gpus.')

parser.add_argument('--max_epis_per_iter', type=int,
                    default=1024, help='Number of episodes in an iteration.')
parser.add_argument('--epoch_per_iter', type=int, default=10,
                    help='Number of epoch in an iteration')
parser.add_argument('--rnn_batch_size', type=int, default=8,
                    help='Number of sequences included in batch of rnn.')
parser.add_argument('--pol_lr', type=float, default=3e-4,
                    help='Policy learning rate')
parser.add_argument('--vf_lr', type=float, default=3e-4,
                    help='Value function learning rate')
parser.add_argument('--cell_size', type=int, default=512,
                    help='Cell size of rnn.')
parser.add_argument('--h_size', type=int, default=512,
                    help='Hidden size of rnn.')

parser.add_argument('--max_grad_norm', type=float, default=0.5,
                    help='Value of maximum gradient norm.')

parser.add_argument('--clip_param', type=float, default=0.2,
                    help='Value of clipping liklihood ratio.')

parser.add_argument('--gamma', type=float, default=0.995,
                    help='Discount factor.')
parser.add_argument('--lam', type=float, default=1,
                    help='Tradeoff value of bias variance.')
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

env1 = GymEnv('HumanoidBulletEnv-v0')
env1.original_env.seed(args.seed)
env1 = AcInObEnv(env1)
env1 = RewInObEnv(env1)
env1 = C2DEnv(env1)

env2 = GymEnv('HumanoidFlagrunBulletEnv-v0')
env2.original_env.seed(args.seed)
env2 = AcInObEnv(env2)
env2 = RewInObEnv(env2)
env2 = C2DEnv(env2)

assert env1.ob_space == env2.ob_space
assert env1.ac_space.shape == env2.ac_space.shape

ob_space = env1.observation_space
ac_space = env1.action_space

pol_net = PolNetLSTM(ob_space, ac_space, h_size=args.h_size,
                     cell_size=args.cell_size)

pol = MultiCategoricalPol(ob_space, ac_space, pol_net,
                          True, data_parallel=args.data_parallel, parallel_dim=1)

vf_net = VNetLSTM(ob_space, h_size=args.h_size, cell_size=args.cell_size)
vf = DeterministicSVfunc(ob_space, vf_net, True,
                         data_parallel=args.data_parallel, parallel_dim=1)

sampler1 = EpiSampler(
    env1, pol, num_parallel=args.num_parallel, seed=args.seed)
sampler2 = EpiSampler(
    env2, pol, num_parallel=args.num_parallel, seed=args.seed)

optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)
optim_vf = torch.optim.Adam(vf_net.parameters(), args.vf_lr)

total_epi = 0
total_step = 0
max_rew = -1e6
while args.max_epis > total_epi:
    with measure('sample'):
        epis1 = sampler1.sample(pol, max_epis=args.max_epis_per_iter)
        epis2 = sampler2.sample(pol, max_epis=args.max_epis_per_iter)
    with measure('train'):
        traj1 = Traj()
        traj2 = Traj()

        traj1.add_epis(epis1)
        traj1 = ef.compute_vs(traj1, vf)
        traj1 = ef.compute_rets(traj1, args.gamma)
        traj1 = ef.compute_advs(traj1, args.gamma, args.lam)
        traj1 = ef.centerize_advs(traj1)
        traj1 = ef.compute_h_masks(traj1)
        traj1.register_epis()

        traj2.add_epis(epis2)
        traj2 = ef.compute_vs(traj2, vf)
        traj2 = ef.compute_rets(traj2, args.gamma)
        traj2 = ef.compute_advs(traj2, args.gamma, args.lam)
        traj2 = ef.centerize_advs(traj2)
        traj2 = ef.compute_h_masks(traj2)
        traj2.register_epis()

        traj1.add_traj(traj2)

        if args.data_parallel:
            pol.dp_run = True
            vf.dp_run = True

        result_dict = ppo_clip.train(traj=traj1, pol=pol, vf=vf, clip_param=args.clip_param,
                                     optim_pol=optim_pol, optim_vf=optim_vf, epoch=args.epoch_per_iter, batch_size=args.batch_size if not args.rnn else args.rnn_batch_size, max_grad_norm=args.max_grad_norm)

        if args.data_parallel:
            pol.dp_run = False
            vf.dp_run = False

    total_epi += traj1.num_epi
    step = traj1.num_step
    total_step += step
    rewards1 = [np.sum(epi['rews']) for epi in epis1]
    rewards2 = [np.sum(epi['rews']) for epi in epis2]
    mean_rew = np.mean(rewards1 + rewards2)
    logger.record_tabular_misc_stat('Reward1', rewards1)
    logger.record_tabular_misc_stat('Reward2', rewards2)
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards1 + rewards2,
                          plot_title='humanoid')

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
    del traj1
    del traj2
del sampler

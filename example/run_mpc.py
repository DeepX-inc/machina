"""
An example of Model Predictive Control.
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
from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol, MPCPol, RandomPol
from machina.algos import mpc
from machina.vfuncs import DeterministicSVfunc
from machina.models import DeterministicSModel
from machina.envs import GymEnv, C2DEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.traj import traj_functional as tf
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import set_device, measure

from simple_net import PolNet, VNet, ModelNet, PolNetLSTM, VNetLSTM, ModelNetLSTM


def add_noise_to_init_obs(epis, std):
    with torch.no_grad():
        for epi in epis:
            epi['obs'][0] += np.random.normal(0, std, epi['obs'][0].shape)
    return epis


def rew_func(next_obs, acs, mean_obs=0., std_obs=1., mean_acs=0., std_acs=1.):
    next_obs = next_obs * std_obs + mean_obs
    acs = acs * std_acs + mean_acs
    # Pendulum
    rews = -(torch.acos(next_obs[:, 0].clamp(min=-1, max=1))**2 +
             0.1*(next_obs[:, 2].clamp(min=-8, max=8)**2) + 0.001 * acs.squeeze(-1)**2)
    rews = rews.squeeze(0)

    return rews


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
parser.add_argument('--num_parallel', type=int, default=4,
                    help='Number of processes to sample.')
parser.add_argument('--cuda', type=int, default=-1, help='cuda device number.')
parser.add_argument('--pybullet_env', action='store_true', default=True)

parser.add_argument('--num_random_rollouts', type=int, default=60,
                    help='Number of random rollouts for collecting initial dataset.')
parser.add_argument('--noise_to_init_obs', type=float, default=0.001,
                    help='Standard deviation of noise to initial observation in initial dataset.')
parser.add_argument('--n_samples', type=int, default=1000,
                    help='Number of samples of action sequence in MPC.')
parser.add_argument('--horizon_of_samples', type=int, default=20,
                    help='Length of horizon of samples of action sequence in MPC.')
parser.add_argument('--max_epis_per_iter', type=int, default=9,
                    help='Number of episodes in an iteration.')
parser.add_argument('--epoch_per_iter', type=int, default=60,
                    help='Number of epochs in an iteration.')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--dm_lr', type=float, default=1e-3)
parser.add_argument('--rnn', action='store_true',
                    default=False, help='If True, network is reccurent.')
parser.add_argument('--rnn_batch_size', type=int, default=8,
                    help='Number of sequences included in batch of rnn.')
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

if args.pybullet_env:
    import pybullet_envs

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

random_pol = RandomPol(observation_space, action_space)

######################
### Model-Based RL ###
######################

### Prepare the dataset D_RAND ###

# Performing rollouts to collect training data
rand_sampler = EpiSampler(
    env, random_pol, num_parallel=args.num_parallel, seed=args.seed)

epis = rand_sampler.sample(random_pol, max_epis=args.num_random_rollouts)
epis = add_noise_to_init_obs(epis, args.noise_to_init_obs)
traj = Traj(traj_device='cpu')
traj.add_epis(epis)
traj = ef.add_next_obs(traj)
traj = ef.compute_h_masks(traj)
# obs, next_obs, and acs should become mean 0, std 1
traj, mean_obs, std_obs, mean_acs, std_acs = ef.normalize_obs_and_acs(traj)
traj.register_epis()

del rand_sampler

### Train Dynamics Model ###

# initialize dynamics model and mpc policy
if args.rnn:
    dm_net = ModelNetLSTM(observation_space, action_space)
else:
    dm_net = ModelNet(observation_space, action_space)
dm = DeterministicSModel(observation_space, action_space, dm_net, args.rnn)
mpc_pol = MPCPol(observation_space, action_space, dm_net, rew_func,
                 args.n_samples, args.horizon_of_samples,
                 mean_obs, std_obs, mean_acs, std_acs, args.rnn)
optim_dm = torch.optim.Adam(dm_net.parameters(), args.dm_lr)

rl_sampler = EpiSampler(
    env, mpc_pol, num_parallel=args.num_parallel, seed=args.seed)

# train loop
total_epi = 0
total_step = 0
counter_agg_iters = 0
max_rew = -1e+6
while args.max_epis > total_epi:
    with measure('train model'):
        result_dict = mpc.train_dm(
            traj, dm, optim_dm, epoch=args.epoch_per_iter, batch_size=args.batch_size if not args.rnn else args.rnn_batch_size)
    with measure('sample'):
        mpc_pol = MPCPol(observation_space, action_space, dm.net, rew_func,
                         args.n_samples, args.horizon_of_samples,
                         mean_obs, std_obs, mean_acs, std_acs, args.rnn)
        epis = rl_sampler.sample(
            mpc_pol, max_epis=args.max_epis_per_iter)

        curr_traj = Traj(traj_device='cpu')
        curr_traj.add_epis(epis)

        curr_traj = ef.add_next_obs(curr_traj)
        curr_traj = ef.compute_h_masks(curr_traj)
        traj = ef.normalize_obs_and_acs(
            curr_traj, mean_obs, std_obs, mean_acs, std_acs, return_statistic=False)
        curr_traj.register_epis()
        traj.add_traj(curr_traj)

    total_epi += curr_traj.num_epi
    step = curr_traj.num_step
    total_step += step
    rewards = [np.sum(epi['rews']) for epi in epis]
    mean_rew = np.mean(rewards)
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

    if mean_rew > max_rew:
        torch.save(dm.state_dict(), os.path.join(
            args.log, 'models', 'dm_max.pkl'))
        torch.save(optim_dm.state_dict(), os.path.join(
            args.log, 'models', 'optim_dm_max.pkl'))
        max_rew = mean_rew

    torch.save(dm.state_dict(), os.path.join(
        args.log, 'models', 'dm_last.pkl'))
    torch.save(optim_dm.state_dict(), os.path.join(
        args.log, 'models', 'optim_dm_last.pkl'))

    del curr_traj
del rl_sampler

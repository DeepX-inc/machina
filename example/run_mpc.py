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
from machina.envs import GymEnv, C2DEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.traj import traj_functional as tf
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import set_device, measure

from simple_net import PolNet, VNet, ModelNet, PolNetLSTM, VNetLSTM


def add_noise_to_init_obs(epis, std):
    with torch.no_grad():
        for epi in epis:
            epi['obs'][0] += np.random.normal(0, std, epi['obs'][0].shape)
    return epis


def rew_func(next_obs, acs):
    # HarfCheetah
    index_of_velx = 3
    rews = next_obs[:, index_of_velx] + 0.05 * \
        torch.sum(acs**2, dim=1, keepdim=True)**2
    rews = rews.squeeze(0)
    return rews


parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage')
parser.add_argument('--env_name', type=str, default='RoboschoolHalfCheetah-v1')
parser.add_argument('--c2d', action='store_true', default=False)
parser.add_argument('--roboschool', action='store_true', default=True)
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--episode', type=int, default=1000000)
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_episodes', type=int, default=1000000)
parser.add_argument('--num_parallel', type=int, default=4)
parser.add_argument('--cuda', type=int, default=-1)

parser.add_argument('--num_rollouts_train', type=int, default=10)
parser.add_argument('--num_rollouts_val', type=int, default=20)
parser.add_argument('--noise_to_init_obs', type=float, default=0.001)
parser.add_argument('--n_samples', type=int, default=1000)
parser.add_argument('--horizon_of_samples', type=int, default=20)
parser.add_argument('--num_aggregation_iters', type=int, default=7)
parser.add_argument('--max_episodes_per_iter', type=int, default=9)
parser.add_argument('--epoch_per_iter', type=int, default=60)
parser.add_argument('--fraction_use_rl_traj', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--dm_lr', type=float, default=1e-4)
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

device_name = 'cpu' if args.cuda < 0 else "cuda:{}".format(args.cuda)
device = torch.device(device_name)
set_device(device)

if args.roboschool:
    import roboschool

score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)

env = GymEnv(args.env_name, log_dir=os.path.join(
    args.log, 'movie'), record_video=args.record)
env.env.seed(args.seed)
if args.c2d:
    env = C2DEnv(env)

ob_space = env.observation_space
ac_space = env.action_space

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

random_pol = RandomPol(ob_space, ac_space)

######################
### Model-Based RL ###
######################

### Prepare the dataset D_RAND ###

# Performing rollouts to collect training data
sampler = EpiSampler(
    env, random_pol, num_parallel=args.num_parallel, seed=args.seed)

epis = sampler.sample(random_pol, max_episodes=args.num_rollouts_train)
epis = add_noise_to_init_obs(epis, args.noise_to_init_obs)
rand_traj = Traj()
rand_traj.add_epis(epis)
rand_traj = ef.add_next_obs(rand_traj)
rand_traj.register_epis()

epis = sampler.sample(random_pol, max_episodes=args.num_rollouts_val)
epis = add_noise_to_init_obs(epis, args.noise_to_init_obs)
rand_traj_val = Traj()
rand_traj_val.add_epis(epis)
rand_traj_val = ef.add_next_obs(rand_traj_val)
rand_traj_val.register_epis()

# obs, next_obs, and acs should become mean 0, std 1
rand_traj, mean_obs, std_obs, mean_acs, std_acs, mean_next_obs, std_next_obs = tf.normalize_obs_and_acs(
    rand_traj)

rl_traj = Traj()

### Train Dynamics Model ###

# initialize dynamics model and mpc policy
dyn_model = ModelNet(ob_space, ac_space)
mpc_pol = MPCPol(ob_space, ac_space, dyn_model, rew_func,
                 args.n_samples, args.horizon_of_samples,
                 mean_obs, std_obs, mean_acs, std_acs,
                 mean_next_obs, std_next_obs)
optim_dm = torch.optim.Adam(dyn_model.parameters(), args.dm_lr)

# train loop
total_epi = 0
total_step = 0
counter_agg_iters = 0
max_rew = -1e-6
while args.num_aggregation_iters > counter_agg_iters:
    with measure('train model'):
        result_dict = mpc.train_dm(
            rl_traj, rand_traj, dyn_model, optim_dm, epoch=args.epoch_per_iter, batch_size=args.batch_size, fraction_use_rl_traj=args.fraction_use_rl_traj)
    with measure('sample'):
        epis = sampler.sample(
            mpc_pol, max_episodes=args.max_episodes_per_iter)

        on_traj = Traj()
        on_traj.add_epis(epis)

        on_traj = ef.add_next_obs(on_traj)
        on_traj.register_epis()
        on_traj = tf.normalize_obs_and_acs(on_traj, mean_obs, std_obs, mean_acs, std_acs,
                                           mean_next_obs, std_next_obs, return_statistic=False)

        rl_traj.add_traj(on_traj)

    total_epi += rl_traj.num_epi
    step = rl_traj.num_step
    total_step += step
    rewards = [np.sum(epi['rews']) for epi in epis]
    mean_rew = np.mean(rewards)
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

    if mean_rew > max_rew:
        torch.save(dyn_model.state_dict(), os.path.join(
            args.log, 'models', 'dm_max.pkl'))
        torch.save(optim_dm.state_dict(), os.path.join(
            args.log, 'models', 'optim_dm_max.pkl'))
        max_rew = mean_rew

    torch.save(dyn_model.state_dict(), os.path.join(
        args.log, 'models', 'dm_last.pkl'))
    torch.save(optim_dm.state_dict(), os.path.join(
        args.log, 'models', 'optim_dm_last.pkl'))

    counter_agg_iters += 1
    del on_traj
del sampler

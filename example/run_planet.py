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
from machina.algos import planet
from machina.pols import RandomPol, PlanetPol
from machina.models import GaussianSModel, RecurrentSSpaceModel
from machina.envs import GymEnv, C2DEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.traj import traj_functional as tf
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import set_device, measure

from simple_net import MLP


def add_noise_to_init_obs(epis, std):
    with torch.no_grad():
        for epi in epis:
            epi['obs'][0] += np.random.normal(0, std, epi['obs'][0].shape)
    return epis


def rew_func(next_obs, acs, mean_obs=0., std_obs=1., mean_acs=0., std_acs=1.):
    next_obs = next_obs * std_obs + mean_obs
    acs = acs * std_acs + mean_acs
    # HarfCheetah
    index_of_velx = 3
    rews = next_obs[:, index_of_velx] - 0.01 * \
        torch.sum(acs**2, dim=1)
    rews = rews.squeeze(0)

    return rews


parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage')
parser.add_argument('--env_name', type=str, default='HalfCheetahBulletEnv-v0')
parser.add_argument('--c2d', action='store_true', default=False)
parser.add_argument('--pybullet_env', action='store_true', default=True)
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_episodes', type=int, default=1000000)
parser.add_argument('--num_parallel', type=int, default=4)
parser.add_argument('--cuda', type=int, default=-1)
parser.add_argument('--data_parallel', action='store_true', default=False)

parser.add_argument('--num_random_rollouts', type=int, default=50)
parser.add_argument('--noise_to_init_obs', type=float, default=0.001)
parser.add_argument('--n_repeat', type=int, default=4)
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--n_samples', type=int, default=1000)
parser.add_argument('--n_refit_samples', type=int, default=100)
parser.add_argument('--n_optim_iters', type=int, default=10)
parser.add_argument('--n_repeat_action', type=int, default=4)
parser.add_argument('--max_episodes_per_iter', type=int, default=1)
parser.add_argument('--epoch_per_iter', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--om_lr', type=float, default=1e-3)
parser.add_argument('--rm_lr', type=float, default=1e-3)
parser.add_argument('--rssm_lr', type=float, default=1e-3)
parser.add_argument('--om_eps', type=float, default=1e-4)
parser.add_argument('--rm_eps', type=float, default=1e-4)
parser.add_argument('--rssm_eps', type=float, default=1e-4)
parser.add_argument('--embed_size', type=int, default=200)
parser.add_argument('--state_size', type=int, default=30)
parser.add_argument('--hidden_size', type=int, default=200)
parser.add_argument('--belief_size', type=int, default=200)
parser.add_argument('--pred_steps_train', type=int, default=50)
parser.add_argument('--max_latend_pred_steps_train', type=int, default=50)
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

if args.pybullet_env:
    import pybullet_envs

score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)

env = GymEnv(args.env_name, log_dir=os.path.join(
    args.log, 'movie'), record_video=args.record)
env.env.seed(args.seed)
if args.c2d:
    env = C2DEnv(env)

ob_space = env.observation_space
ac_space = env.action_space

random_pol = RandomPol(ob_space, ac_space)

### Prepare the random traj ###

# Performing rollouts to collect training data
rand_sampler = EpiSampler(
    env, random_pol, num_parallel=args.num_parallel, seed=args.seed, n_repeat=args.n_repeat)

epis = rand_sampler.sample(random_pol, max_episodes=args.num_random_rollouts)
epis = add_noise_to_init_obs(epis, args.noise_to_init_obs)
traj = Traj()
traj.add_epis(epis)
traj = ef.add_next_obs(traj)
traj = ef.compute_h_masks(traj)
traj.register_epis()

del rand_sampler

### Init Models ###

# initialize dynamics model and mpc policy
ob_net = MLP(args.state_size + args.belief_size,
             args.embed_size, deterministic=False)
rew_net = MLP(args.state_size + args.belief_size, 1, deterministic=False)
ob_model = GaussianSModel(ob_space, ac_space, ob_net,
                          rnn=False, data_parallel=args.data_parallel)
rew_model = GaussianSModel(ob_space, ac_space, rew_net,
                           rnn=False, data_parallel=args.data_parallel)
rssm = RecurrentSSpaceModel(ob_space, ac_space, args.embed_size, args.state_size, args.belief_size, args.hidden_size,
                            data_parallel=args.data_parallel, parallel_dim=1)
optim_om = torch.optim.Adam(ob_model.parameters(), args.om_lr, eps=args.om_eps)
optim_rm = torch.optim.Adam(rew_model.parameters(),
                            args.rm_lr, eps=args.rm_eps)
optim_rssm = torch.optim.Adam(
    rssm.parameters(), args.rssm_lr, eps=args.rssm_eps)

planet_pol = PlanetPol(ob_space, ac_space, rssm, rew_model, args.horizon,
                       args.n_optim_iters, args.n_samples, args.n_refit_samples,
                       data_parallel=False, parallel_dim=0)
rl_sampler = EpiSampler(
    env, planet_pol, num_parallel=args.num_parallel, seed=args.seed, n_repeat=args.n_repeat)

### Train Models ###

# train loop
total_epi = 0
total_step = 0
counter_agg_iters = 0
max_rew = -1e+6
while args.max_episodes > total_epi:
    with measure('train'):
        result_dict = planet.train(
            traj, rssm, ob_model, rew_model, optim_rssm, optim_om, optim_rm, epoch=args.epoch_per_iter,
            pred_steps=args.pred_steps_train, max_latend_pred_steps=args.max_latend_pred_steps_train,
            batch_size=args.batch_size, num_epi_per_seq=1)
    """
    total_epi += 1
    step = 1
    total_step += 1
    rewards = [1]
    """
    with measure('sample'):
        planet_pol = PlanetPol(ob_space, ac_space, rssm, rew_model, args.horizon,
                               args.n_optim_iters, args.n_samples, args.n_refit_samples,
                               data_parallel=False, parallel_dim=0)
        epis = rl_sampler.sample(
            planet_pol, max_episodes=args.max_episodes_per_iter)

        curr_traj = Traj()
        curr_traj.add_epis(epis)

        curr_traj = ef.add_next_obs(curr_traj)
        curr_traj = ef.compute_h_masks(curr_traj)
        curr_traj.register_epis()
        traj.add_traj(curr_traj)

    total_epi += curr_traj.num_epi
    step = curr_traj.num_step
    total_step += step
    rewards = [np.sum(epi['rews']) for epi in epis]

    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)
    counter_agg_iters += 1
    #del curr_traj

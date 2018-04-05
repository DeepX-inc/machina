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
import pybullet_envs

import machina as mc
from machina.pols import GaussianPol, MixtureGaussianPol
from machina.algos import svg
from machina.prepro import BasePrePro
from machina.qfuncs import DeterministicQfunc
from machina.envs import GymEnv
from machina.data import ReplayData, GAEData
from machina.samplers import BatchSampler
from machina.misc import logger
from machina.utils import set_gpu, measure
from net import PolNet, QNet, MixturePolNet

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage')
parser.add_argument('--env_name', type=str, default='LunarLanderContinuous-v2')
parser.add_argument('--roboschool', action='store_true', default=False)
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--episode', type=int, default=1000000)
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_episodes', type=int, default=1000000)
parser.add_argument('--cuda', type=int, default=-1)

parser.add_argument('--mixture', type=int, default=1)
parser.add_argument('--max_data_size', type=int, default=1000000)
parser.add_argument('--min_data_size', type=int, default=10000)
parser.add_argument('--max_samples_per_iter', type=int, default=2000)
parser.add_argument('--max_episodes_per_iter', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--sampling', type=int, default=10)
parser.add_argument('--pol_lr', type=float, default=1e-4)
parser.add_argument('--qf_lr', type=float, default=3e-4)
parser.add_argument('--use_prepro', action='store_true', default=False)

parser.add_argument('--batch_type', type=str, choices=['large', 'small'], default='large')

parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--gamma', type=float, default=0.99)
args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available() else -1
set_gpu(args.cuda)

if not os.path.exists(args.log):
    os.mkdir(args.log)

with open(os.path.join(args.log, 'args.json'), 'w') as f:
    json.dump(vars(args), f)
pprint(vars(args))

if not os.path.exists(os.path.join(args.log, 'models')):
    os.mkdir(os.path.join(args.log, 'models'))

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.roboschool:
    import roboschool

score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)

env = GymEnv(args.env_name, log_dir=os.path.join(args.log, 'movie'), record_video=args.record)
env.env.seed(args.seed)

ob_space = env.observation_space
ac_space = env.action_space

if args.mixture > 1:
    pol_net = MixturePolNet(ob_space, ac_space, args.mixture)
    pol = MixtureGaussianPol(ob_space, ac_space, pol_net)
else:
    pol_net = PolNet(ob_space, ac_space)
    pol = GaussianPol(ob_space, ac_space, pol_net)
qf_net = QNet(ob_space, ac_space)
qf = DeterministicQfunc(ob_space, ac_space, qf_net)
targ_qf = copy.deepcopy(qf)
prepro = BasePrePro(ob_space)
sampler = BatchSampler(env)
optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)
optim_qf = torch.optim.Adam(qf_net.parameters(), args.qf_lr)

total_epi = 0
total_step = 0
max_rew = -1e6
off_data = ReplayData(args.max_data_size, ob_space.shape[0], ac_space.shape[0])
while args.max_episodes > total_epi:
    with measure('sample'):
        if args.use_prepro:
            paths = sampler.sample(pol, args.max_samples_per_iter, args.max_episodes_per_iter, prepro.prepro_with_update)
        else:
            paths = sampler.sample(pol, args.max_samples_per_iter, args.max_episodes_per_iter)

    total_epi += len(paths)
    step = sum([len(path['rews']) for path in paths])
    total_step += step

    off_data.add_paths(paths)

    if off_data.size <= args.min_data_size:
        continue

    with measure('train'):
        result_dict = svg.train(
            off_data,
            pol, qf, targ_qf,
            optim_pol,optim_qf, step, args.batch_size,
            args.tau, args.gamma, args.sampling,
        )

    rewards = [np.sum(path['rews']) for path in paths]
    mean_rew = np.mean(rewards)
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

    mean_rew = np.mean([np.sum(path['rews']) for path in paths])
    if mean_rew > max_rew:
        torch.save(pol.state_dict(), os.path.join(args.log, 'models', 'pol_max.pkl'))
        torch.save(qf.state_dict(), os.path.join(args.log, 'models', 'qf_max.pkl'))
        torch.save(optim_pol.state_dict(), os.path.join(args.log, 'models', 'optim_pol_max.pkl'))
        torch.save(optim_qf.state_dict(), os.path.join(args.log, 'models', 'optim_qf_max.pkl'))
        max_rew = mean_rew

    torch.save(pol.state_dict(), os.path.join(args.log, 'models', 'pol_last.pkl'))
    torch.save(qf.state_dict(), os.path.join(args.log, 'models', 'qf_last.pkl'))
    torch.save(optim_pol.state_dict(), os.path.join(args.log, 'models', 'optim_pol_last.pkl'))
    torch.save(optim_qf.state_dict(), os.path.join(args.log, 'models', 'optim_qf_last.pkl'))






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
from machina.pols import GaussianPol
from machina.algos import sac
from machina.prepro import BasePrePro
from machina.vfuncs import DeterministicVfunc
from machina.qfuncs import DeterministicQfunc
from machina.envs import GymEnv
from machina.data import ReplayData
from machina.samplers import BatchSampler
from machina.misc import logger
from machina.utils import set_gpu, measure
from net import PolNet, QNet, VNet

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage')
parser.add_argument('--env_name', type=str, default='LunarLanderContinuous-v2')
parser.add_argument('--roboschool', action='store_true', default=False)
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--episode', type=int, default=1000000)
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_episodes', type=int, default=1000000)
parser.add_argument('--cuda', type=int, default=-1)

parser.add_argument('--max_data_size', type=int, default=1000000)
parser.add_argument('--min_data_size', type=int, default=10000)
parser.add_argument('--max_samples_per_iter', type=int, default=2000)
parser.add_argument('--max_episodes_per_iter', type=int, default=10000)
parser.add_argument('--update_per_step', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--sampling', type=int, default=10)
parser.add_argument('--pol_lr', type=float, default=1e-4)
parser.add_argument('--qf_lr', type=float, default=3e-4)
parser.add_argument('--vf_lr', type=float, default=3e-4)
parser.add_argument('--use_prepro', action='store_true', default=False)

parser.add_argument('--ent_alpha', type=float, default=1)
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

logger.add_tabular_output(os.path.join(args.log, 'progress.csv'))

env = GymEnv(args.env_name, log_dir=os.path.join(args.log, 'movie'), record_video=args.record)
env.env.seed(args.seed)

ob_space = env.observation_space
ac_space = env.action_space

pol_net = PolNet(ob_space, ac_space)
pol = GaussianPol(ob_space, ac_space, pol_net)
qf_net = QNet(ob_space, ac_space)
qf = DeterministicQfunc(ob_space, ac_space, qf_net)
vf_net = VNet(ob_space)
vf = DeterministicVfunc(ob_space, vf_net)
prepro = BasePrePro(ob_space)
sampler = BatchSampler(env)
optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)
optim_qf = torch.optim.Adam(qf_net.parameters(), args.qf_lr)
optim_vf = torch.optim.Adam(vf_net.parameters(), args.vf_lr)

total_epi = 0
total_step = 0
max_rew = -1e6
off_data = ReplayData(args.max_data_size, ob_space.shape[0], ac_space.shape[0], rew_scale=1/args.ent_alpha)
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
        result_dict = sac.train(
            off_data,
            pol, qf, vf,
            optim_pol,optim_qf, optim_vf,
            step * args.update_per_step, args.batch_size,
            args.gamma, args.sampling,
        )

    for key, value in result_dict.items():
        if not hasattr(value, '__len__'):
            logger.record_tabular(key, value)
        elif len(value) >= 1:
            logger.record_tabular_misc_stat(key, value)
    logger.record_tabular('PolLogStd', pol_net.log_std_param.data.cpu().numpy()[0])
    logger.record_tabular_misc_stat('Reward', [np.sum(path['rews']) for path in paths])
    logger.record_tabular('EpisodePerIter', len(paths))
    logger.record_tabular('TotalEpisode', total_epi)
    logger.record_tabular('StepPerIter', step)
    logger.record_tabular('TotalStep', total_step)
    logger.dump_tabular()

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



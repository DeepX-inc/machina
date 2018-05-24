import argparse
import copy
import json
import os
from pprint import pprint
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from gym.wrappers.monitoring import Monitor
import os
from machina.pols import DeterministicPol
from example.net import DeterministicPolNet, DeterministicPolNetBN
from machina.utils import Variable, torch2torch, np2torch
import mujoco_py

from machina.samplers import BatchSampler
from machina.misc import logger
from machina.utils import set_gpu, measure

from machina.algos import behavior_cloning
from machina.data import ExpertData


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='/home/yoshida/research/movie/models/')
parser.add_argument('--file_name', type=str, default='halfcheetah_rllab_3232_noBN_0202pol_max.pkl')
parser.add_argument('--hidden_layer1', type=int, default=32)
parser.add_argument('--hidden_layer2', type=int, default=32)
parser.add_argument('--expert_dir', type = str, default='machina/data/expert_data_npz')
parser.add_argument('--expert_path', type = str, default='halfcheetah_rllab_3232_noBN_0202pol_max_HalfCheetah-v1_30trajs.npz')

parser.add_argument('--log', type=str, default='garbage')
parser.add_argument('--log_filename', type=str, default='')
parser.add_argument('--env_name', type=str, default='HalfCheetah-v30')
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--episode', type=int, default=1000000)
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--env_seed', type=int, default=256)
parser.add_argument('--max_episodes', type=int, default=1000000)

parser.add_argument('--max_data_size', type=int, default=1000000)
parser.add_argument('--min_data_size', type=int, default=10000)
parser.add_argument('--max_samples_per_iter', type=int, default=10000)
parser.add_argument('--max_episodes_per_iter', type=int, default=10000)
parser.add_argument('--performance_check_per_epoch', type=float, default=1e-1)

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--pol_lr', type=float, default=1e-3)
parser.add_argument('--cuda', type=int, default=-1)

parser.add_argument('--batch_normalization', action='store_true', default=False)
parser.add_argument('--apply_noise', action='store_true', default=False)

parser.add_argument('--reuse', action='store_true', default=False)
parser.add_argument('--reuse_filename', type=str, default='')
parser.add_argument('--num_of_step', type=int, default=1)
parser.add_argument('--num_of_traj', type=int, default=30)
parser.add_argument('--weight_decay', type=float, default=0.0)


args = parser.parse_args()


filename = args.log_filename + '{}expert_trajs'.format(args.num_of_traj)

env = gym.make(args.env_name)
ob_space = env.observation_space
ac_space = env.action_space
#env=Monitor(env, os.path.join('HalfCheetahMovie'), force=True)


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

logger.add_tabular_output(os.path.join(args.log, filename+'progress.csv'))

env.env.seed(args.env_seed)

if args.batch_normalization:
    pol_net = DeterministicPolNetBN(ob_space, ac_space, args.hidden_layer1, args.hidden_layer2)
else:
    pol_net = DeterministicPolNet(ob_space, ac_space, args.hidden_layer1, args.hidden_layer2)

pol = DeterministicPol(ob_space, ac_space, pol_net, None, None)

if args.reuse:
    with open(os.path.join(args.log, 'models',args.reuse_filename +  'pol_last.pkl'), 'rb') as f:
        pol.load_state_dict(torch.load(f))
optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)
if args.reuse:
    with open(os.path.join(args.log, 'models',args.reuse_filename +  'optim_pol_last.pkl'), 'rb') as f:
        optim_pol.load_state_dict(torch.load(f))

if args.reuse:
    with open(os.path.join(args.log, 'models',args.reuse_filename +  'pol_last.pkl'), 'rb') as f:
        pol.load_state_dict(torch.load(f))


sampler = BatchSampler(env)
expert_data = ExpertData(os.path.join(os.getcwd(),args.expert_dir, args.expert_path))


total_epi = 0
total_step = 0
max_rew = -1e6


for current_epoch in range(args.epoch):
    with measure('training_time_per_epoch'):
        result_dict = behavior_cloning.train(
            expert_data, pol, optim_pol,
            args.epoch, args.batch_size
        )

    logger.record_tabular('Epoch', current_epoch)
    for key, value in result_dict.items():
        if not hasattr(value, '__len__'):
            logger.record_tabular(key, value)
        elif len(value) >= 1:
            logger.record_tabular_misc_stat(key, value)

    if current_epoch % int(args.epoch*args.performance_check_per_epoch)==0:
        pol.eval()
        paths = sampler.sample(pol, args.max_samples_per_iter, args.max_episodes_per_iter)
        pol.train()
        total_epi += len(paths)
        step = sum([len(path['rews']) for path in paths])
        total_step += step
        current_reward = [np.sum(path['rews']) for path in paths]
        mean_rew = np.mean([np.sum(path['rews']) for path in paths])
        if mean_rew > max_rew:
            torch.save(pol.state_dict(), os.path.join(args.log, 'models', filename + 'pol_max.pkl'))
            torch.save(optim_pol.state_dict(), os.path.join(args.log, 'models', filename + 'optim_pol_max.pkl'))
            max_rew = mean_rew

        torch.save(pol.state_dict(), os.path.join(args.log, 'models', filename + 'pol_last.pkl'))
        torch.save(optim_pol.state_dict(), os.path.join(args.log, 'models', filename + 'optim_pol_last.pkl'))

    logger.record_tabular_misc_stat('Reward', current_reward)
    logger.record_tabular('EpisodePerIter', len(paths))
    logger.record_tabular('TotalEpisode', total_epi)
    logger.record_tabular('StepPerIter', step)
    logger.record_tabular('TotalStep', total_step)
    logger.dump_tabular()


"""
Distributed (multi-GPU) training example using torch.DDP.
Use torch.distributed.launch to start this script.
Only the rank 0 perform sampling.

Example:
- 1node, 2GPU
    python -m torch.distributed.launch \
        --nproc_per_node 2 \
        --master_addr 192.168.10.1 \
        --master_port 12341 \
        --nnode 1 \
        --node_rank 0 \
        ./run_ppo_distributed.py
"""

import argparse
import json
import os
from pprint import pprint

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import gym
import pybullet_envs

import machina as mc
from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol
from machina.algos import ppo_clip, ppo_kl
from machina.vfuncs import DeterministicSVfunc
from machina.envs import GymEnv, C2DEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.traj import traj_functional as tf
from machina.samplers.raysampler import EpiSampler
from machina import logger
from machina.utils import measure, set_device, make_redis, init_ray, make_model_distributed

from simple_net import PolNet, VNet, PolNetLSTM, VNetLSTM

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage',
                    help='Directory name of log.')
parser.add_argument('--env_name', type=str,
                    default='AntBulletEnv-v0', help='Name of environment.')
parser.add_argument('--c2d', action='store_true',
                    default=False, help='If True, action is discretized.')
parser.add_argument('--record', action='store_true',
                    default=False, help='If True, movie is saved.')
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_epis', type=int,
                    default=1000000, help='Number of episodes to run.')
parser.add_argument('--num_parallel', type=int, default=4,
                    help='Number of processes to sample.')
parser.add_argument('--redis_host', type=str, default="localhost",
                    help='hostname where redis server is launched.')
parser.add_argument('--redis_port', type=str, default="6379",
                    help='port number for redis.')

# DDP option
parser.add_argument('--local_rank', type=int,
                    help='Local rank of this process. This option is given by torch.distributed.launch.')
parser.add_argument('--backend', type=str, default='nccl',
                    choices=['nccl', 'gloo', 'mpi'],
                    help='backend of torch.distributed.')
parser.add_argument('--master_address', type=str,
                    default='tcp://127.0.0.1:12389',
                    help='address that belongs to the rank 0 process.')
parser.add_argument('--use_apex', action="store_true",
                    help='if True, use nvidia/apex insatead of torch.DDP.')
parser.add_argument('--apex_opt_level', type=str, default="O0",
                    help='apex option. optimization level.')
parser.add_argument('--apex_keep_batchnorm_fp32', type=bool, default=None,
                    help='apex option. keep batch norm weights in fp32.')
parser.add_argument('--apex_loss_scale', type=float, default=None,
                    help='apex option. loss scale.')
parser.add_argument('--apex_sync_bn', action="store_true",
                    help='apex option. sync batch norm statistics.')

# Ray option (ray is used for sampling)
parser.add_argument('--ray_redis_address', type=str, default=None,
                    help='Ray cluster\'s address that this programm connect to. If not specified, start ray locally.')

parser.add_argument('--max_steps_per_iter', type=int, default=100000,
                    help='Number of steps to use in an iteration.')
parser.add_argument('--epoch_per_iter', type=int, default=10,
                    help='Number of epoch in an iteration')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--pol_lr', type=float, default=3e-4,
                    help='Policy learning rate')
parser.add_argument('--vf_lr', type=float, default=3e-4,
                    help='Value function learning rate')

parser.add_argument('--rnn', action='store_true',
                    default=False, help='If True, network is reccurent.')
parser.add_argument('--rnn_batch_size', type=int, default=8,
                    help='Number of sequences included in batch of rnn.')
parser.add_argument('--max_grad_norm', type=float, default=10,
                    help='Value of maximum gradient norm.')

parser.add_argument('--ppo_type', type=str,
                    choices=['clip', 'kl'], default='clip', help='Type of Proximal Policy Optimization.')

parser.add_argument('--clip_param', type=float, default=0.2,
                    help='Value of clipping liklihood ratio.')

parser.add_argument('--kl_targ', type=float, default=0.01,
                    help='Target value of kl divergence.')
parser.add_argument('--init_kl_beta', type=float,
                    default=1, help='Initial kl coefficient.')

parser.add_argument('--gamma', type=float, default=0.99,
                    help='Discount factor.')
parser.add_argument('--lam', type=float, default=0.95,
                    help='Tradeoff value of bias variance.')
args = parser.parse_args()


if not os.path.exists(args.log):
    os.makedirs(args.log)

dist.init_process_group(backend=args.backend, init_method="env://")
rank = dist.get_rank()

if rank == 0:
    init_ray(ray_redis_address=args.ray_redis_address)
    with open(os.path.join(args.log, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
    pprint(vars(args))

make_redis(args.redis_host, args.redis_port)

if not os.path.exists(os.path.join(args.log, 'models')):
    os.mkdir(os.path.join(args.log, 'models'))

np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = args.local_rank

device_name = "cuda:{}".format(args.cuda)
device = torch.device(device_name)
set_device(device)

score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)
logger.add_tensorboard_output(args.log)

env = GymEnv(args.env_name)
env.env.seed(args.seed)
if args.c2d:
    env = C2DEnv(env)

observation_space = env.observation_space
action_space = env.action_space

if args.rnn:
    pol_net = PolNetLSTM(observation_space, action_space,
                         h_size=256, cell_size=256)
else:
    pol_net = PolNet(observation_space, action_space)
if isinstance(action_space, gym.spaces.Box):
    pol = GaussianPol(observation_space, action_space, pol_net, args.rnn)
elif isinstance(action_space, gym.spaces.Discrete):
    pol = CategoricalPol(observation_space, action_space, pol_net, args.rnn)
elif isinstance(action_space, gym.spaces.MultiDiscrete):
    pol = MultiCategoricalPol(
        observation_space, action_space, pol_net, args.rnn)
else:
    raise ValueError('Only Box, Discrete, and MultiDiscrete are supported')

if args.rnn:
    vf_net = VNetLSTM(observation_space, h_size=256, cell_size=256)
else:
    vf_net = VNet(observation_space)
vf = DeterministicSVfunc(observation_space, vf_net, args.rnn)

if rank == 0:
    sampler = EpiSampler(
        env, pol, num_parallel=args.num_parallel, seed=args.seed)

optim_pol = torch.optim.Adam(pol.parameters(), args.pol_lr)
optim_vf = torch.optim.Adam(vf.parameters(), args.vf_lr)

ddp_pol, optim_pol = make_model_distributed(pol, optim_pol,
                                            args.use_apex, args.apex_opt_level,
                                            args.apex_keep_batchnorm_fp32, args.apex_sync_bn,
                                            args.apex_loss_scale,
                                            device_ids=[args.local_rank],
                                            output_device=args.local_rank)
ddp_vf, optim_vf = make_model_distributed(vf, optim_vf,
                                          args.use_apex, args.apex_opt_level,
                                          args.apex_keep_batchnorm_fp32, args.apex_sync_bn,
                                          args.apex_loss_scale,
                                          device_ids=[args.local_rank],
                                          output_device=args.local_rank)

total_epi = 0
total_step = 0
max_rew = -1e6
kl_beta = args.init_kl_beta
while args.max_epis > total_epi:
    with measure('sample', log_enable=rank == 0):
        if rank == 0:
            epis = sampler.sample(pol, max_steps=args.max_steps_per_iter)
    with measure('train', log_enable=rank == 0):
        traj = Traj(ddp=True, traj_device="cpu")
        if rank == 0:
            traj.add_epis(epis)

            traj = ef.compute_vs(traj, vf)
            traj = ef.compute_rets(traj, args.gamma)
            traj = ef.compute_advs(traj, args.gamma, args.lam)
            traj = ef.centerize_advs(traj)
            traj = ef.compute_h_masks(traj)
            traj.register_epis()
        traj = tf.sync(traj)

        if args.ppo_type == 'clip':
            result_dict = ppo_clip.train(traj=traj, pol=ddp_pol, vf=ddp_vf, clip_param=args.clip_param,
                                         optim_pol=optim_pol,
                                         optim_vf=optim_vf,
                                         epoch=args.epoch_per_iter,
                                         batch_size=args.batch_size if not
                                         args.rnn else args.rnn_batch_size,
                                         max_grad_norm=args.max_grad_norm,
                                         log_enable=rank == 0)
        else:
            result_dict = ppo_kl.train(traj=traj, pol=ddp_pol, vf=ddp_vf, kl_beta=kl_beta, kl_targ=args.kl_targ,
                                       optim_pol=optim_pol, optim_vf=optim_vf,
                                       epoch=args.epoch_per_iter,
                                       batch_size=args.batch_size if not
                                       args.rnn else args.rnn_batch_size,
                                       max_grad_norm=args.max_grad_norm,
                                       log_enable=rank == 0)
            kl_beta = result_dict['new_kl_beta']

    total_epi += traj.num_epi
    step = traj.num_step
    total_step += step
    if rank == 0:
        rewards = [np.sum(epi['rews']) for epi in epis]
        mean_rew = np.mean(rewards)
        logger.record_results(args.log, result_dict, score_file,
                              total_epi, step, total_step,
                              rewards,
                              plot_title=args.env_name)

    if rank == 0:
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
    del traj
if rank == 0:
    del sampler

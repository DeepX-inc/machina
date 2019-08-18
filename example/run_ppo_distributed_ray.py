"""
Distributed (multi-GPU) training example (ray + DDP version).

This script use ray to manage all processes. No need to use
torch.distributed.launch to start this program.

For a single node, this script automatically sets up ray.
To use multi nodes, first you need to set up ray cluster.
For example, run `ray start --head --redis-port 58300 --node-ip-address 192.168.10.1`
on node 1 and run `ray start --redis-address 192.168.10.1:58300` on node 2.
For more information see https://ray.readthedocs.io/en/latest/using-ray-on-a-cluster.html

Example:
- single node, 1 trainer (1GPU), 8 samplers:
    - python run_ppo_distributed.py --trainer 1 --num_parallel 8
- single node, 2 trainer (2GPU), 8 samplers:
    - python run_ppo_distributed.py --trainer 2 --num_parallel 8
- multi node, 1 trainer (1GPU), 20 samplers:
    - python run_ppo_distributed.py --trainer 1 --num_parallel 20 --ray_redis_address <ray_cluster_addr>

Program overview:

           Main loop
|-------------------------------|
|     init()                    |
|       |                       |
|  |--->|                       |
|  |    |                       |
|  |  sample() -> EpiSampler    |
|  |    |                       |
|  |    |                       |
|  |  train() -> Trainer (DDP)  |
|  |    |                       |
|  |    |                       |
|  |<---|                       |
|-------------------------------|
"""

import argparse
import copy
import json
import os
from pprint import pprint
import time
import types

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import gym
import pybullet_envs
import ray

from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol
from machina.algos import ppo_clip
from machina.vfuncs import DeterministicSVfunc
from machina.envs import GymEnv, C2DEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.traj import traj_functional as tf
from machina.samplers.raysampler import EpiSampler
from machina import logger
from machina.utils import measure, init_ray
from machina.utils import make_model_distributed, BaseDistributedRayTrainer, TrainManager

from simple_net import PolNet, VNet, PolNetLSTM, VNetLSTM


class Trainer(BaseDistributedRayTrainer):
    """GPU distributed trainer using torch.DDP
      NOTE:
       - Internally this has two type of networks, one is the original and the other is ddp version,
         i.e., pol and ddp_pol / vf and ddp_vf. Both network share the parameters.
         Original network is for serialization (i.e., get_state()).
         (note that ddp_pol.module == pol).
      """

    def __init__(self, args, pol, vf, rank=0, world_size=1, master_address=None, backend="nccl"):
        super(Trainer, self).__init__(rank, world_size,
                                      master_address, backend, args.seed)
        self.args = args
        self.rank = rank

        self.pol = pol.to(self.device)
        self.vf = vf.to(self.device)
        self.pol.train()
        self.vf.train()

        # NOTE: We need to create optimizer in the trainer because we need to
        # pass the parameters on the GPU
        self.optim_pol = torch.optim.Adam(
            self.pol.parameters(), self.args.pol_lr)
        self.optim_vf = torch.optim.Adam(self.vf.parameters(), self.args.vf_lr)

        self.ddp_pol, self.optim_pol = make_model_distributed(self.pol, self.optim_pol,
                                                              args.use_apex, args.apex_opt_level,
                                                              args.apex_keep_batchnorm_fp32, args.apex_sync_bn,
                                                              args.apex_loss_scale)
        self.ddp_vf, self.optim_vf = make_model_distributed(self.vf, self.optim_vf,
                                                            args.use_apex, args.apex_opt_level,
                                                            args.apex_keep_batchnorm_fp32, args.apex_sync_bn,
                                                            args.apex_loss_scale)

    def train(self, epis):
        traj = Traj(ddp=True, traj_device=self.device)
        traj.add_epis(epis)

        traj = ef.compute_vs(traj, self.vf)
        traj = ef.compute_rets(traj, args.gamma)
        traj = ef.compute_advs(traj, args.gamma, args.lam)
        traj = ef.centerize_advs(traj)
        traj = ef.compute_h_masks(traj)
        traj.register_epis()

        result_dict = ppo_clip.train(traj=traj, pol=self.ddp_pol, vf=self.ddp_vf,
                                     clip_param=self.args.clip_param,
                                     optim_pol=self.optim_pol, optim_vf=self.optim_vf,
                                     epoch=self.args.epoch_per_iter,
                                     batch_size=self.args.batch_size,
                                     max_grad_norm=self.args.max_grad_norm,
                                     log_enable=self.rank == 0)

        result_dict["traj_num_step"] = traj.num_step
        result_dict["traj_num_epi"] = traj.num_epi
        return result_dict


def main(args):
    init_ray(args.num_cpus, args.num_gpus, args.ray_redis_address)

    if not os.path.exists(args.log):
        os.makedirs(args.log)
    if not os.path.exists(os.path.join(args.log, 'models')):
        os.mkdir(os.path.join(args.log, 'models'))
    score_file = os.path.join(args.log, 'progress.csv')
    logger.add_tabular_output(score_file)
    logger.add_tensorboard_output(args.log)
    with open(os.path.join(args.log, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
    pprint(vars(args))

    # when doing the distributed training, disable video recordings
    env = GymEnv(args.env_name)
    env.env.seed(args.seed)
    if args.c2d:
        env = C2DEnv(env)

    observation_space = env.observation_space
    action_space = env.action_space
    pol_net = PolNet(observation_space, action_space)
    rnn = False
    # pol_net = PolNetLSTM(observation_space, action_space)
    # rnn = True
    if isinstance(action_space, gym.spaces.Box):
        pol = GaussianPol(observation_space,
                          action_space, pol_net, rnn=rnn)
    elif isinstance(action_space, gym.spaces.Discrete):
        pol = CategoricalPol(observation_space, action_space, pol_net)
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        pol = MultiCategoricalPol(observation_space, action_space, pol_net)
    else:
        raise ValueError(
            'Only Box, Discrete, and MultiDiscrete are supported')

    vf_net = VNet(observation_space)
    vf = DeterministicSVfunc(observation_space, vf_net)

    trainer = TrainManager(Trainer, args.num_trainer, args.master_address,
                           args=args, vf=vf, pol=pol)
    sampler = EpiSampler(env, pol,
                         args.num_parallel, seed=args.seed)

    total_epi = 0
    total_step = 0
    max_rew = -1e6
    start_time = time.time()

    while args.max_epis > total_epi:

        with measure('sample'):
            sampler.set_pol_state(trainer.get_state("pol"))
            epis = sampler.sample(max_steps=args.max_steps_per_iter)

        with measure('train'):
            result_dict = trainer.train(epis=epis)

        step = result_dict["traj_num_step"]
        total_step += step
        total_epi += result_dict["traj_num_epi"]

        rewards = [np.sum(epi['rews']) for epi in epis]
        mean_rew = np.mean(rewards)
        elapsed_time = time.time() - start_time
        logger.record_tabular('ElapsedTime', elapsed_time)
        logger.record_results(args.log, result_dict, score_file,
                              total_epi, step, total_step,
                              rewards,
                              plot_title=args.env_name)

        with measure('save'):
            pol_state = trainer.get_state("pol")
            vf_state = trainer.get_state("vf")
            optim_pol_state = trainer.get_state("optim_pol")
            optim_vf_state = trainer.get_state("optim_vf")

            torch.save(pol_state, os.path.join(
                args.log, 'models', 'pol_last.pkl'))
            torch.save(vf_state, os.path.join(
                args.log, 'models', 'vf_last.pkl'))
            torch.save(optim_pol_state, os.path.join(
                args.log, 'models', 'optim_pol_last.pkl'))
            torch.save(optim_vf_state, os.path.join(
                args.log, 'models', 'optim_vf_last.pkl'))

            if mean_rew > max_rew:
                torch.save(pol_state, os.path.join(
                    args.log, 'models', 'pol_max.pkl'))
                torch.save(vf_state, os.path.join(
                    args.log, 'models', 'vf_max.pkl'))
                torch.save(optim_pol_state, os.path.join(
                    args.log, 'models', 'optim_pol_max.pkl'))
                torch.save(optim_vf_state, os.path.join(
                    args.log, 'models', 'optim_vf_max.pkl'))
                max_rew = mean_rew
    del sampler
    del trainer


if __name__ == "__main__":
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

    parser.add_argument('--max_steps_per_iter', type=int, default=100000,
                        help='Number of steps to use in an iteration.')
    parser.add_argument('--epoch_per_iter', type=int, default=10,
                        help='Number of epoch in an iteration')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pol_lr', type=float, default=3e-4,
                        help='Policy learning rate')
    parser.add_argument('--vf_lr', type=float, default=3e-4,
                        help='Value function learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=10,
                        help='Value of maximum gradient norm.')
    parser.add_argument('--ppo_type', type=str,
                        choices=['clip', 'kl'], default='clip', help='Type of Proximal Policy Optimization.')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='Value of clipping liklihood ratio.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor.')
    parser.add_argument('--lam', type=float, default=0.95,
                        help='Tradeoff value of bias variance.')

    # DDP option
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

    # Ray option
    parser.add_argument('--ray_redis_address', type=str, default=None,
                        help='Ray cluster\'s address that this programm connect to. If not specified, start ray locally.')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs that ray manages. Only effective when launching ray locally. default: all GPUs available.')
    parser.add_argument('--num_cpus', type=int, default=None,
                        help='Number of CPUs that ray manages. Only effective when launching ray locally. default: all CPUs available.')
    parser.add_argument('--num_trainer', type=int, default=1,
                        help='Number of trainers (number of GPUs to train).')
    args = parser.parse_args()

    main(args)

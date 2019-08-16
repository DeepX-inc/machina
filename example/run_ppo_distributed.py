"""
An example of distributed training
You can do
  1) Multi-GPU trainig (using torch.DDP)
  2) Multi-CPU parallel sampling (using ray)

This script use ray for distributed computations.
For a single node, this script automatically sets up ray.
To use multi nodes, first you need to set up ray cluster.
For example, run `ray start --head --redis-port 58300 --node-ip-address 192.168.10.1`
on node 1 and run `ray start --redis-address 192.168.10.1:58300` on node 2.
For more information see https://ray.readthedocs.io/en/latest/using-ray-on-a-cluster.html

Example:
- single node, 1 trainer (1GPU), 8 samplers:
    - python run_ppo_distributed.py --trainer 1 --num_sample_workers 8
- single node, 2 trainer (2GPU), 8 samplers:
    - python run_ppo_distributed.py --trainer 2 --num_sample_workers 8
- multi node, 1 trainer (1GPU), 20 samplers:
    - python run_ppo_distributed.py --trainer 1 --num_sample_workers 20 --ray_redis_address <ray_cluster_addr>

Program overview:

        Agent (main loop)
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

NOTE:
- Use ray for launching processes and distributed epi sampling
- Ray launches worker processes of EpiSampler and Trainer
- Use torch.DDP for distributed GPU training
- Use 1GPU per trainer
- Use CPUs for epi sampling (no GPU is used)
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
from machina.utils import measure, set_device, wrap_ddp, init_ray

from simple_net import PolNet, VNet, PolNetLSTM, VNetLSTM


@ray.remote(num_gpus=1)
class Trainer:
    """GPU distributed trainer using torch.DDP
    NOTE:
     - Ray sets CUDA_VISIBLE_DEVICES for this worker. So use "cuda:0" for the device.
     - Internally this has two type of networks, one is the original and the other is ddp version,
       i.e., pol and ddp_pol / vf and ddp_vf. Both network share the parameters.
       Original network is for serialization (e.g., get_pol_state()).
       (note that ddp_pol.module == pol).
    """

    def __init__(self, args, observation_space, action_space, rank, world_size, master_address, backend="nccl"):
        self.args = args

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.pol = self._make_pol(
            observation_space, action_space)
        self.vf = self._make_vf(observation_space)
        self.optim_pol, self.optim_vf = self._make_optimizer()

        # XXX: serialization of apex model fails for some reason. As a
        # workaround, copy original model and use it for the model
        # serialization. See get_pol() and get_pol_state() function below.
        self.orig_pol = copy.deepcopy(self.pol)

        self.device = f"cuda:0"
        set_device(self.device)
        torch.cuda.set_device(0)

        self.pol.to(self.device)
        self.vf.to(self.device)

        # Initialize DDP
        dist.init_process_group(
            backend=backend, init_method=master_address, world_size=world_size, rank=rank)

        if args.use_apex:
            global amp
            global apex
            import apex.parallel
            from apex import amp
            self.ddp_pol, self.optim_pol = amp.initialize(self.pol, self.optim_pol,
                                                          opt_level=self.args.apex_opt_level,
                                                          keep_batchnorm_fp32=self.args.apex_keep_batchnorm_fp32,
                                                          loss_scale=self.args.apex_loss_scale)
            self.ddp_vf, self.optim_vf = amp.initialize(self.vf, self.optim_vf,
                                                        opt_level=self.args.apex_opt_level,
                                                        keep_batchnorm_fp32=self.args.apex_keep_batchnorm_fp32,
                                                        loss_scale=self.args.apex_loss_scale)
            ddp_cls = wrap_ddp(apex.parallel.DistributedDataParallel)
            self.ddp_pol = ddp_cls(self.ddp_pol)
            self.ddp_vf = ddp_cls(self.ddp_vf)
            if self.args.apex_sync_bn:
                self.ddp_pol = apex.parallel.convert_syncbn_model(self.ddp_pol)
                self.ddp_vf = apex.parallel.convert_syncbn_model(self.ddp_vf)
        else:
            ddp_cls = wrap_ddp(nn.parallel.DistributedDataParallel)
            self.ddp_pol = ddp_cls(self.pol, device_ids=[self.device], dim=0)
            self.ddp_vf = ddp_cls(self.vf, device_ids=[self.device], dim=0)

    def _make_pol(self, observation_space, action_space):
        pol_net = PolNet(observation_space, action_space)
        rnn = False
        #pol_net = PolNetLSTM(observation_space, action_space)
        #rnn = True
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
        return pol

    def _make_vf(self, observation_space):
        vf_net = VNet(observation_space)
        vf = DeterministicSVfunc(observation_space, vf_net)
        return vf

    def _make_optimizer(self):
        optim_pol = torch.optim.Adam(self.pol.parameters(), self.args.pol_lr)
        optim_vf = torch.optim.Adam(self.vf.parameters(), self.args.vf_lr)
        return optim_pol, optim_vf

    def _to_cpu(self, state_dict):
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.to("cpu")
            if isinstance(v, dict):
                state_dict[k] = self._to_cpu(v)
        return state_dict

    def get_pol(self):
        """Return pol network
        XXX: This is for getting pol definition. Use get_pol_state() to get the current weights.
        """
        return self.orig_pol

    def get_pol_state(self):
        return self._to_cpu(copy.deepcopy(self.pol.state_dict()))

    def get_vf_state(self):
        return self._to_cpu(copy.deepcopy(self.vf.state_dict()))

    def get_optim_pol_state(self):
        return self._to_cpu(copy.deepcopy(self.optim_pol.state_dict()))

    def get_optim_vf_state(self):
        return self._to_cpu(copy.deepcopy(self.optim_vf.state_dict()))

    def train(self, epis, return_result=False):
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
                                     log_enable=return_result)
        if return_result:
            result_dict["traj_num_step"] = traj.num_step
            result_dict["traj_num_epi"] = traj.num_epi
            return result_dict


class Agent:
    def __init__(self, args):
        self.args = args

        self.score_file = self._init_log()
        self.env = self._make_env()

        self.trainers = self._make_trainer()
        self.sampler = self._make_sampler()

    def _init_log(self):
        if not os.path.exists(self.args.log):
            os.makedirs(self.args.log)
        if not os.path.exists(os.path.join(self.args.log, 'models')):
            os.mkdir(os.path.join(self.args.log, 'models'))
        score_file = os.path.join(args.log, 'progress.csv')
        logger.add_tabular_output(score_file)
        logger.add_tensorboard_output(args.log)
        with open(os.path.join(args.log, 'args.json'), 'w') as f:
            json.dump(vars(args), f)
        pprint(vars(args))
        return score_file

    def _make_env(self):
        # when doing the distributed training, disable video recordings
        # env = GymEnv(args.env_name, log_dir=os.path.abspath(os.path.join(
        #     args.log, 'movie')), record_video=args.record)
        env = GymEnv(args.env_name)
        env.env.seed(args.seed)
        if self.args.c2d:
            env = C2DEnv(env)
        return env

    def _make_sampler(self):
        pol = ray.get(self.trainers[0].get_pol.remote())
        sampler = EpiSampler(self.env, pol,
                             args.num_sample_workers, seed=self.args.seed)
        return sampler

    def _make_trainer(self):
        trainers = [Trainer.remote(args, self.env.observation_space,
                                   self.env.action_space,
                                   rank=i, world_size=args.num_trainer,
                                   master_address=args.master_address) for i in range(args.num_trainer)]
        return trainers

    def log(self, msg, nl=True):
        print(msg, end='\n' if nl else '')

    def main(self):
        total_epi = 0
        total_step = 0
        max_rew = -1e6
        start_time = time.time()

        self.log("training start")
        while self.args.max_epis > total_epi:

            with measure('sample'):
                self.sampler.set_pol_state(
                    self.trainers[0].get_pol_state.remote())
                epis = self.sampler.sample(max_steps=args.max_steps_per_iter)

            epis_obj = ray.put(epis)
            with measure('train'):
                results = [t.train.remote(epis_obj, return_result=i == 0) for i, t in
                           enumerate(self.trainers)]
                results = ray.get(results)

            result_dict = results[0]
            step = result_dict["traj_num_step"]
            total_step += step
            total_epi += result_dict["traj_num_epi"]

            rewards = [np.sum(epi['rews']) for epi in epis]
            mean_rew = np.mean(rewards)
            elapsed_time = time.time() - start_time
            logger.record_tabular('ElapsedTime', elapsed_time)
            logger.record_results(args.log, result_dict, self.score_file,
                                  total_epi, step, total_step,
                                  rewards,
                                  plot_title=args.env_name)

            with measure('save'):
                pol_state = ray.get(self.trainers[0].get_pol_state.remote())
                vf_state = ray.get(self.trainers[0].get_vf_state.remote())
                optim_pol_state = ray.get(
                    self.trainers[0].get_optim_pol_state.remote())
                optim_vf_state = ray.get(
                    self.trainers[0].get_vf_state.remote())

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


def main(args):
    init_ray(args.num_cpus, args.num_gpus, args.ray_redis_address)
    cluster_resources = ray.cluster_resources()
    print(f"Ray cluster resources: {cluster_resources}")
    assert args.num_trainer <= cluster_resources['GPU']

    agent = Agent(args)
    agent.main()


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
    parser.add_argument('--num_sample_workers', type=int, default=4,
                        help='Number of processes to sample.')
    parser.add_argument('--num_envs', type=int, default=1,
                        help='number of envs per sample worker')
    parser.add_argument('--num_batch_epi', type=int, default=1,
                        help='number of episodes to sample at a time')
    parser.add_argument('--cuda', type=int, default=-
                        1, help='cuda device number.')

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
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--master_address', type=str,
                        default='tcp://127.0.0.1:12389')
    parser.add_argument('--use_apex', action="store_true")
    parser.add_argument('--apex_opt_level', type=str, default="O0")
    parser.add_argument('--apex_keep_batchnorm_fp32', type=bool, default=None)
    parser.add_argument('--apex_loss_scale', type=float, default=None)
    parser.add_argument('--apex_sync_bn', action="store_true")

    # Ray option
    parser.add_argument('--ray_redis_address', type=str, default=None)
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--num_cpus', type=int, default=None)
    parser.add_argument('--num_trainer', type=int, default=1)
    args = parser.parse_args()

    main(args)

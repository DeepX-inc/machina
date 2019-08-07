"""
An Example of Diversity Is All You Need.
https://arxiv.org/abs/1802.06070

"""

import os
import gym
import torch
import argparse
import numpy as np
import pybullet_envs

import torch
import torch.nn as nn
import torch.nn.functional as F

from machina import logger
from machina.traj import Traj
from machina.envs import SkillEnv
from machina.utils import measure
from machina.algos import diayn_sac, diayn
from machina.pols import GaussianPol
from machina.samplers import EpiSampler
from machina.utils import set_device, measure
from machina.traj import epi_functional as ef
from machina.vfuncs import DeterministicSAVfunc, DeterministicSVfunc
from simple_net import PolNet, QNet, DiaynDiscrimNet

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage',
                    help='Directory name of log.')
parser.add_argument('--env_name', type=str,
                    default='ReacherBulletEnv-v0', help='Name of environment.')
parser.add_argument('--cuda', type=int, default=-1, help='cuda device number.')
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_episodes', type=int, default=1e8,
                    help='Number of episodes to run.')
parser.add_argument('--num_parallel', type=int, default=4,
                    help='Number of processes to sample.')
parser.add_argument('--max_steps_per_iter', type=int, default=2000,
                    help='Number of steps to use in an iteration.')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--discrim_batch_size', type=int, default=32)
parser.add_argument('--discrim_h_size', type=int,
                    default=100, help='Hidden size of Discriminator.')
parser.add_argument('--pol_lr', type=float, default=1e-4,
                    help='Policy learning rate.')
parser.add_argument('--qf_lr', type=float, default=3e-4,
                    help='Q function learning rate.')
parser.add_argument('--discrim_lr', type=float,
                    default=0.001, help='Discriminator learning rate.')
parser.add_argument('--discrim_momentum', type=float,
                    default=0.9, help='Discriminator momentum.')
parser.add_argument('--epoch_per_iter', type=int,
                    default=100, help='Number of epoch in an iteration.')
parser.add_argument('--gamma', type=float,
                    default=0.99, help='Discount factor.')
parser.add_argument('--tau', type=float, default=5e-3,
                    help='Coefficient of target function.')
parser.add_argument('--sampling', type=int, default=1,
                    help='Number of sampling in calculation of expectation.')
parser.add_argument('--no_reparam', action='store_true', default=False)
parser.add_argument('--steps_per_save_models', type=float,
                    default=2e5, help='Frequency of saving model per steps.')
parser.add_argument('--num_skill', type=int, default=4,
                    help='Number of skills.')
args = parser.parse_args()


'''
Feature extractor of the discriminator.
please see the paper 4.4.2 question 7.
e.g.:
def discrim_f(x): return x
f_dim = 9  # dimension of feature space
'''


def discrim_f(x): return x[:, 0:2]+x[:, 2:4]


f_dim = 2


env = gym.make(args.env_name)
env = SkillEnv(env, num_skill=args.num_skill)
obs = env.reset()
observation_space = env.real_observation_space
skill_space = env.skill_space
ob_skill_space = env.observation_space
action_space = env.action_space
ob_dim = ob_skill_space.shape[0] - args.num_skill
device_name = 'cpu' if args.cuda < 0 else "cuda:{}".format(args.cuda)
device = torch.device(device_name)
set_device(device)

# policy
pol_net = PolNet(ob_skill_space, action_space)
pol = GaussianPol(ob_skill_space, action_space, pol_net)

# q-function
qf_net1 = QNet(ob_skill_space, action_space)
qf1 = DeterministicSAVfunc(ob_skill_space, action_space, qf_net1)
targ_qf_net1 = QNet(ob_skill_space, action_space)
targ_qf_net1.load_state_dict(qf_net1.state_dict())
targ_qf1 = DeterministicSAVfunc(ob_skill_space, action_space, targ_qf_net1)
qf_net2 = QNet(ob_skill_space, action_space)
qf2 = DeterministicSAVfunc(ob_skill_space, action_space, qf_net2)
targ_qf_net2 = QNet(ob_skill_space, action_space)
targ_qf_net2.load_state_dict(qf_net2.state_dict())
targ_qf2 = DeterministicSAVfunc(ob_skill_space, action_space, targ_qf_net2)
qfs = [qf1, qf2]
targ_qfs = [targ_qf1, targ_qf2]

log_alpha = nn.Parameter(torch.ones((), device=device))

high = np.array([np.finfo(np.float32).max]*f_dim)
f_space = gym.spaces.Box(-high, high, dtype=np.float32)
discrim_net = DiaynDiscrimNet(
    f_space, skill_space, h_size=args.discrim_h_size, discrim_f=discrim_f).to(device)

discrim = DeterministicSVfunc(f_space, discrim_net, rnn=False)


# set optimizer to both models
optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)
optim_qf1 = torch.optim.Adam(qf_net1.parameters(), args.qf_lr)
optim_qf2 = torch.optim.Adam(qf_net2.parameters(), args.qf_lr)
optim_qfs = [optim_qf1, optim_qf2]
optim_alpha = torch.optim.Adam([log_alpha], args.pol_lr)
optim_discrim = torch.optim.SGD(discrim.parameters(
), lr=args.discrim_lr, momentum=args.discrim_momentum)

off_traj = Traj()
sampler = EpiSampler(
    env, pol, num_parallel=args.num_parallel, seed=args.seed)

if not os.path.exists(args.log):
    os.makedirs(args.log)
    os.makedirs(args.log+'/models')
score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)
logger.add_tensorboard_output(args.log)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# counter and record for loop
total_epi = 0
total_step = 0
mean_rew = 0

discrim_loss = []
discrim_total_step = []

# train loop
while args.max_episodes > total_epi:
    print('totalepi:', total_epi, 'mean_rew', mean_rew)
    # sample trajectories
    with measure('sample'):
        epis = sampler.sample(pol, max_steps=args.max_steps_per_iter)

    with measure('train'):
        on_traj = Traj()
        on_traj.add_epis(epis)

        on_traj = ef.add_next_obs(on_traj)
        on_traj = ef.compute_diayn_rews(
            on_traj, lambda x: diayn_sac.calc_rewards(x, args.num_skill, discrim))
        on_traj.register_epis()
        off_traj.add_traj(on_traj)

        total_epi += on_traj.num_epi
        step = on_traj.num_step
        total_step += step
        log_alpha = nn.Parameter(
            np.log(0.1)*torch.ones((), device=device))  # fix alpha

        result_dict = diayn_sac.train(
            off_traj,
            pol, qfs, targ_qfs, log_alpha,
            optim_pol, optim_qfs, optim_alpha,
            step, args.batch_size,
            args.tau, args.gamma, args.sampling,
            discrim, args.num_skill,
            not args.no_reparam)
        discrim_losses = diayn.train(discrim, optim_discrim, on_traj,
                                     args.discrim_batch_size, args.epoch_per_iter,
                                     args.num_skill)
    # update counter and record
    rewards = [np.sum(epi['rews']) for epi in epis]
    result_dict['discrimloss'] = discrim_losses
    mean_rew = np.mean(rewards)
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

    # save models regular intervals
    steps_as = str(int(
        int(total_step / args.steps_per_save_models + 1) * args.steps_per_save_models))
    if 'prev_as' in locals():
        if not prev_as == steps_as:
            torch.save(pol.state_dict(), os.path.join(
                args.log, 'models', 'pol_'+steps_as+'.pkl'))
            torch.save(qf1.state_dict(), os.path.join(
                args.log, 'models', 'qf1_'+steps_as+'.pkl'))
            torch.save(qf2.state_dict(), os.path.join(
                args.log, 'models', 'qf2_'+steps_as+'.pkl'))
            torch.save(discrim.state_dict(), os.path.join(
                args.log, 'models', 'discrim_'+steps_as+'.pkl'))
            torch.save(optim_pol.state_dict(), os.path.join(
                args.log, 'models', 'optim_pol_'+steps_as+'.pkl'))
            torch.save(optim_qf1.state_dict(), os.path.join(
                args.log, 'models', 'optim_qf1_'+steps_as+'.pkl'))
            torch.save(optim_qf2.state_dict(), os.path.join(
                args.log, 'models', 'optim_qf2_'+steps_as+'.pkl'))
            torch.save(optim_discrim.state_dict(), os.path.join(
                args.log, 'models', 'optim_discrim_'+steps_as+'.pkl'))
    prev_as = str(int(
        int(total_step / args.steps_per_save_models + 1) * args.steps_per_save_models))
    del on_traj
del sampler

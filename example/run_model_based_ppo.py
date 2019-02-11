"""
An example of Proximal Policy Gradient.
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
from machina.pols import GaussianPol, CategoricalPol, MultiCategoricalPol
from machina.algos import model_based_ppo
from machina.vfuncs import DeterministicSVfunc
from machina.envs import GymEnv, C2DEnv, ImaginaryEnv
from machina.traj import Traj
from machina.traj import epi_functional as ef
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import measure, set_device

from simple_net import PolNet, VNet, PolNetLSTM, VNetLSTM, ModelNet

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage')
parser.add_argument('--env_name', type=str, default='Pendulum-v0')
parser.add_argument('--c2d', action='store_true', default=False)
parser.add_argument('--roboschool', action='store_true', default=False)
parser.add_argument('--record', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_episodes', type=int, default=1000000)
parser.add_argument('--num_parallel', type=int, default=4)

parser.add_argument('--max_steps_per_iter', type=int, default=10000)
parser.add_argument('--epoch_per_iter', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--pol_lr', type=float, default=1e-4)
parser.add_argument('--vf_lr', type=float, default=3e-4)
parser.add_argument('--ob_model_lr', type=float, default=1e-4)
parser.add_argument('--rew_model_lr', type=float, default=1e-4)
parser.add_argument('--cuda', type=int, default=-1)

parser.add_argument('--rnn', action='store_true', default=False)
parser.add_argument('--max_grad_norm', type=float, default=10)

parser.add_argument('--ppo_type', type=str,
                    choices=['clip', 'kl'], default='clip')

parser.add_argument('--clip_param', type=float, default=0.2)

parser.add_argument('--kl_targ', type=float, default=0.01)
parser.add_argument('--init_kl_beta', type=float, default=1)

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

if args.rnn:
    vf_net = VNetLSTM(ob_space, h_size=256, cell_size=256)
else:
    vf_net = VNet(ob_space)
vf = DeterministicSVfunc(ob_space, vf_net, args.rnn)

ob_model = ModelNet(ob_space, ac_space, ob_space.shape[0])
rew_model = ModelNet(ob_space, ac_space, 1)
imaginary_env = ImaginaryEnv(args.env_name, ob_model, rew_model)

sampler = EpiSampler(env, pol, num_parallel=args.num_parallel, seed=args.seed)
imaginary_sampler = EpiSampler(imaginary_env, pol, num_parallel=args.num_parallel, seed=args.seed+1)

optim_pol = torch.optim.Adam(pol_net.parameters(), args.pol_lr)
optim_vf = torch.optim.Adam(vf_net.parameters(), args.vf_lr)
optim_ob_model = torch.optim.SGD(ob_model.parameters(), args.ob_model_lr)
optim_rew_model = torch.optim.SGD(rew_model.parameters(), args.rew_model_lr)

total_epi = 0
total_step = 0
max_rew = -1e6
kl_beta = args.init_kl_beta
off_traj = Traj()

while args.max_episodes > total_epi:
    with measure('sample trajectries'):
        epis = sampler.sample(pol, max_steps=args.max_steps_per_iter)
    with measure('train models'):
        on_traj = Traj()
        on_traj.add_epis(epis)
        on_traj = ef.add_next_obs(on_traj)
        on_traj.register_epis()
        off_traj.add_traj(on_traj)
        step = on_traj.num_step

        result_dict = model_based_ppo.train_models(traj=off_traj, ob_model=ob_model, rew_model=rew_model,
                                         optim_ob_model=optim_ob_model, optim_rew_model=optim_rew_model,
                                         epoch=step, batch_size=args.batch_size)
    with measure('sample imaginary trajectries'):
        del imaginary_sampler
        imaginary_env = ImaginaryEnv(args.env_name, ob_model, rew_model)
        imaginary_sampler = EpiSampler(imaginary_env, pol, num_parallel=args.num_parallel, seed=np.random.randint(0,256))
        imaginary_epis = imaginary_sampler.sample(pol, max_steps=args.max_steps_per_iter)
    with measure('train policy and vf'):
        im_traj = Traj()
        im_traj.add_epis(imaginary_epis)

        im_traj = ef.compute_vs(im_traj, vf)
        im_traj = ef.compute_rets(im_traj, args.gamma)
        im_traj = ef.compute_advs(im_traj, args.gamma, args.lam)
        im_traj = ef.centerize_advs(im_traj)
        im_traj = ef.compute_h_masks(im_traj)
        im_traj.register_epis()

        im_result_dict = model_based_ppo.train_policy_and_vf(traj=im_traj, pol=pol, vf=vf, clip_param=args.clip_param,
                                            optim_pol=optim_pol, optim_vf=optim_vf, epoch=args.epoch_per_iter, batch_size=args.batch_size, max_grad_norm=args.max_grad_norm)
    total_epi += on_traj.num_epi
    step = on_traj.num_step
    total_step += step
    rewards = [np.sum(epi['rews']) for epi in epis]
    mean_rew = np.mean(rewards)
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

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
    del im_traj, on_traj
del sampler
import torch
import argparse
import pickle
import gym
from gym.wrappers.monitoring import Monitor


import os
from machina.pols import DeterministicPol, GaussianPol
from machina.nets import DeterministicPolNet, PolNet
import mujoco_py
import numpy as np
from machina.utils import measure, set_device
from machina.envs import GymEnv

parser = argparse.ArgumentParser()
parser.add_argument('--from_dir', type=str, default='expert_traj/policy_model_file')
parser.add_argument('--file_name', type=str, default='pol_max.pkl')
parser.add_argument('--to_dir', type=str, default='expert_traj/expert_traj_file')
parser.add_argument('--h1', type=int, default=32)
parser.add_argument('--h2', type=int, default=32)
parser.add_argument('--num_of_traj', type=int, default=100)
parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
parser.add_argument('--seed', type=int, default='256')
parser.add_argument('--cuda', type=int, default='-1')
parser.add_argument('--gaussian_pol', action='store_true', default=False)
args = parser.parse_args()

env = GymEnv(args.env_name, record_video=False)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.env.seed(args.seed)
set_device(args.cuda)

ob_space = env.observation_space
ac_space = env.action_space

if args.gaussian_pol:
    pol_net = PolNet(ob_space, ac_space, args.h1, args.h2)
    pol = GaussianPol(ob_space, ac_space, pol_net)
else:
    pol_net = DeterministicPolNet(ob_space, ac_space, args.h1, args.h2)
    pol = DeterministicPol(ob_space, ac_space, pol_net)

with open(os.path.join(args.from_dir,args.file_name), 'rb') as f:
    pol.load_state_dict(torch.load(f, map_location=lambda storage, location: storage))

obs = []
acs = []
rews = []
dones = []
rets = []
lens = []
for i in range(args.num_of_traj):
    obs = []
    acs = []
    rews = []
    dones = []
    ob = env.reset()
    done = False
    reward = 0

    cur_ep_ret = 0
    cur_ep_len = 0
    while not done:
        obs.append(ob)
        dones.append(int(done))
        action_real, _, _ = pol.deterministic_ac_real(torch.tensor(ob, dtype=torch.float).unsqueeze(0))
        acs.append(action_real[0])
        ob, reward, done, info = env.step(action_real[0])
        rews.append(reward)
        cur_ep_ret += reward
        cur_ep_len += 1
    rets.append(cur_ep_ret)
    lens.append(cur_ep_len)

filename = env.spec.id + '_{}trajs'.format(args.num_of_traj)
np.savez(os.path.join(args.to_dir, filename), obs=obs, acs=acs, dones=dones,
         lens=lens, rets=rets)
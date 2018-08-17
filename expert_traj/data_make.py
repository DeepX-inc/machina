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
parser.add_argument('--file_name', type=str, default='pol_max.pkl')
parser.add_argument('--h1', type=int, default=32)
parser.add_argument('--h2', type=int, default=32)
parser.add_argument('--num_of_traj', type=int, default=100)
parser.add_argument('--env_name', type=str, default='HalfCheetah-v1')
parser.add_argument('--seed', type=int, default='256')
parser.add_argument('--cuda', type=int, default='-1')
parser.add_argument('--stochastic', action='store_true', default=False)
args = parser.parse_args()

env = GymEnv(args.env_name, log_dir=os.path.join(args.log, 'movie'), record_video=args.record)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.env.seed(args.seed)
set_device(args.cuda)

ob_space = env.observation_space
ac_space = env.action_space

if args.stochastic:
    pol_net = DeterministicPolNet(ob_space, ac_space, args.hidden_layer1, args.hidden_layer2)
    pol = DeterministicPol(ob_space, ac_space, pol_net, None, None)
else:
    pol_net = DeterministicPolNet(ob_space, ac_space, args.hidden_layer1, args.hidden_layer2)
    pol = DeterministicPol(ob_space, ac_space, pol_net, None, None)

with open(os.getcwd()+ '/' + args.dir + args.file_name, 'rb') as f:
    pol.load_state_dict(torch.load(f, map_location=lambda storage, location: storage))

obs_list = []
acs_list = []
len_list = []
ret_list = []
num_of_traj = args.num_of_traj
num_of_timestep = 1000
for i in range(num_of_traj):
 #   np.random.seed(i)
 #   torch.manual_seed(i)
    obs = []
    acs = []
    rews = []
    env.seed(i+1)
    ob = env.reset()
#    print(ob[-3:])
    done = False
    reward = 0
    t = 0

    cur_ep_ret = 0
    cur_ep_len = 0
    while True:
        action_real, _, _ = pol((torch.from_numpy(ob).float().unsqueeze(0)))
        obs.append(ob)
        acs.append(action_real[0])
        ob, reward, done, info = env.step(action_real[0])
        t += 1
        rews.append(reward)
        cur_ep_ret += reward
        cur_ep_len += 1
        if done:
            print('current_ep_len:{}'.format(cur_ep_len))
            break
#        env.render()
#    print(t)
    obs_list.append(np.array(obs))
    acs_list.append(np.array(acs))
    ret_list.append(cur_ep_ret)
    len_list.append(cur_ep_len)


filename = args.file_name.split('.')[0] + '_' + env.spec.id + '_{}trajs'.format(num_of_traj)

np.savez(os.path.join(os.getcwd(),'suggest_method', 'expert_data', filename), obs=np.array(obs_list), acs=np.array(acs_list),
         lens=np.array(len_list), rets=np.array(ret_list))
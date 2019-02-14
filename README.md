<div align="center"><img src="assets/machina_logo.jpg" width="800"/></div>

<br />
<br />

[![Build Status](https://travis-ci.com/DeepX-inc/machina.svg?token=xZEqXwSaqc7xZ2saWZa2&branch=master)](https://travis-ci.com/DeepX-inc/machina)
[![Python Version](https://img.shields.io/pypi/pyversions/Django.svg)](https://github.com/DeepX-inc/machina)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/DeepX-inc/machina/blob/master/LICENSE)

# machina

machina is a library for real-world Deep Reinforcement Learning and which is built on top of PyTorch.

## Features
+ Composability
  The sampling phase and learning phase are independent. They interact with each other only via the Policy and Trajectory.

Using the principle of composability, we can easily implement following configurations which are difficult in other RL libraries.
1. An agent learns in mixed environment (e.g. simulated environment and real world environment, some meta learning settings).
2. An agent learns by multiple algorithms rather than a single algorithm (e.g. Q-Prop is combination of TRPO and DDPG).
3. Hyperparameters for an algorithm are changing dynamically (e.g. Meta Learning).

### 1 Meta Reinforcement Learning example
We usually define meta learning as a fast adaptation method for tasks which is sampled from task-space.
In meta RL, task is defined as MDP.
RL agent have to adapt new MDP quickly.
We have to sample episodes from different environments to learn a meta agent.
We can implement this easily like below with machina.

```python:run_mixed_env.py
env1 = GymEnv('HumanoidBulletEnv-v0')

env2 = GymEnv('HumanoidFlagrunBulletEnv-v0')

epis1 = sampler1.sample(pol, max_episodes=args.max_episodes_per_iter)
epis2 = sampler2.sample(pol, max_episodes=args.max_episodes_per_iter)
traj1 = Traj()
traj2 = Traj()

traj1.add_epis(epis1)
traj1 = ef.compute_vs(traj1, vf)
traj1 = ef.compute_rets(traj1, args.gamma)
traj1 = ef.compute_advs(traj1, args.gamma, args.lam)
traj1 = ef.centerize_advs(traj1)
traj1 = ef.compute_h_masks(traj1)
traj1.register_epis()

traj2.add_epis(epis2)
traj2 = ef.compute_vs(traj2, vf)
traj2 = ef.compute_rets(traj2, args.gamma)
traj2 = ef.compute_advs(traj2, args.gamma, args.lam)
traj2 = ef.centerize_advs(traj2)
traj2 = ef.compute_h_masks(traj2)
traj2.register_epis()

traj1.add_traj(traj2)

result_dict = ppo_clip.train(traj=traj1, pol=pol, vf=vf, clip_param=args.clip_param,
                             optim_pol=optim_pol, optim_vf=optim_vf, epoch=args.epoch_per_iter, batch_size=args.batch_size, max_grad_norm=args.max_grad_norm)
```

You can see full of this code [here].

### 2 Combination of Off-policy and On-policy algorithms
DeepRL algorithms can be roughly divided into 2 types.
On-policy and Off-policy algorithms.
On-policy algorithms use only current episodes for updating policy or some value functions.
On the other hand, Off-policy algorithms use whole episodes for updating policy or some value functions.
On-policy algorithms are stable but need many episdeos.
Off-policy algorithms are sample efficient but unstable.
Some algorithms like [Q-Prop](https://arxiv.org/abs/1611.02247) are combination of On-policy and Off-policy algorithms.
This is an example of the combination of [ppo](https://arxiv.org/abs/1707.06347) and [sac](https://arxiv.org/abs/1801.01290).

```
epis = sampler.sample(pol, max_steps=args.max_steps_per_iter)

on_traj = Traj()
on_traj.add_epis(epis)

on_traj = ef.add_next_obs(on_traj)
on_traj = ef.compute_vs(on_traj, vf)
on_traj = ef.compute_rets(on_traj, args.gamma)
on_traj = ef.compute_advs(on_traj, args.gamma, args.lam)
on_traj = ef.centerize_advs(on_traj)
on_traj = ef.compute_h_masks(on_traj)
on_traj.register_epis()

result_dict1 = ppo_clip.train(traj=on_traj, pol=pol, vf=vf, clip_param=args.clip_param,
                            optim_pol=optim_pol, optim_vf=optim_vf, epoch=args.epoch_per_iter, batch_size=args.batch_size, max_grad_norm=args.max_grad_norm)

total_epi += on_traj.num_epi
step = on_traj.num_step
total_step += step

off_traj.add_traj(on_traj)

result_dict2 = sac.train(
    off_traj,
    pol, qf, targ_qf, log_alpha,
    optim_pol, optim_qf, optim_alpha,
    100, args.batch_size,
    args.tau, args.gamma, args.sampling,
)
```

You can see full code here.

To obtain this composability, machina's sampling method is restricted to be episode-based. Episode-based sampling is suitable for real-world environments. Some algorithms which update networks step by step (e.g. DQN, DDPG) are not reproduced in machina.


## Installation

machina supports Python3.5, 3.6 and PyTorch0.4+.

machina can be installed using PyPI.
```
pip install machina
```

Or you can install machina directly from source code.
```
git clone https://github.com/DeepX-inc/machina
cd machina
python setup.py install
```

## Quick Start
Let's start machina with a [quickstart](https://github.com/DeepX-inc/machina/tree/master/example/quickstart).

You can check some implemented algorithms in [examples](https://github.com/DeepX-inc/machina/tree/master/example) as a first step.



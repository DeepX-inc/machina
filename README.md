<div align="center"><img src="assets/machina_logo.jpg" width="800"/></div>

<br />
<br />

[![Build Status](https://travis-ci.com/DeepX-inc/machina.svg?token=xZEqXwSaqc7xZ2saWZa2&branch=master)](https://travis-ci.com/DeepX-inc/machina)
[![Python Version](https://img.shields.io/pypi/pyversions/Django.svg)](https://github.com/DeepX-inc/machina)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/DeepX-inc/machina/blob/master/LICENSE)

# machina

machina is a library for real-world Deep Reinforcement Learning which is built on top of PyTorch.

## Features
+ Composability
  The sampling phase and learning phase are completely independent. They interact with each other only via policy and trajectories sampled.

Using the principle of composability, we can easily implement following configurations which are otherwise difficult in other RL libraries.
1. An agent which learns in a mixed environment (e.g. simulated environment and real world environment, some meta learning settings).
2. An agent which learns by multiple algorithms rather than just a single one (e.g. Q-Prop is combination of TRPO and DDPG).
3. Hyperparameters for an algorithm are changing dynamically (e.g. Meta Learning).

### 1 Meta Reinforcement Learning example
We usually define meta learning as a fast adaptation method for tasks which are sampled from a task-space.
In meta RL, a task is defined as a MDP.
RL agents have to adapt to a new MDP as fast as possible.
We have to sample episodes from different environments to train our meta agent.
We can easily implement this like below with machina.

```python:run_mixed_env.py
env1 = GymEnv('HumanoidBulletEnv-v0')

env2 = GymEnv('HumanoidFlagrunBulletEnv-v0')

epis1 = sampler1.sample(pol, max_epis=args.max_epis_per_iter)
epis2 = sampler2.sample(pol, max_epis=args.max_epis_per_iter)
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

You can see the full example code [here].

### 2 Combination of Off-policy and On-policy algorithms
DeepRL algorithms can be roughly divided into 2 types.
On-policy and Off-policy algorithms.
On-policy algorithms use only current episodes for updating policy or some value functions.
On the other hand, Off-policy algorithms use whole episodes for updating policy or some value functions.
On-policy algorithms are more stable but need many episodeos.
Off-policy algorithms are sample efficient but unstable.
Some algorithms like [Q-Prop](https://arxiv.org/abs/1611.02247) are a combination of On-policy and Off-policy algorithms.
This is an example of the combination using [ppo](https://arxiv.org/abs/1707.06347) and [sac](https://arxiv.org/abs/1801.01290).

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

You can see the full example code here.

To obtain this composability, machina's sampling method is deliberatly restricted to be episode-based because episode-based sampling is suitable for real-world environments. Moreover, some algorithms which update networks step by step (e.g. DQN, DDPG) are not reproduced in machina.


## Implemented Algorithms

+ [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
+ [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
+ [Soft Actor Critic](https://arxiv.org/abs/1801.01290)
+ [QT-Opt](https://arxiv.org/abs/1806.10293)
+ [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971)
+ [Stochastic Value Gradient](https://arxiv.org/abs/1510.09142)
+ [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)
+ [Adversatial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248)
+ Behavior Cloning
+ [Model-Based RL (Model Predictive Control)](https://arxiv.org/abs/1708.02596)
+ [Teacher Distillation](https://arxiv.org/abs/1902.02186)


## Installation

machina supports Ubuntu, Python3.5, 3.6, 3.7 and PyTorch1.0.0+.

machina can be directly installed using PyPI.
```
pip install machina
```

Or you can install machina from source code.
```
git clone https://github.com/DeepX-inc/machina
cd machina
python setup.py install
```

## Quick Start
You can start machina by checking out this [quickstart](https://github.com/DeepX-inc/machina/tree/master/example/quickstart).

Moreover, you can also check already implemented algorithms in [examples](https://github.com/DeepX-inc/machina/tree/master/example).


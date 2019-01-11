<div align="center"><img src="assets/machina_logo.jpg" width="800"/></div>

# machina

Machina is a library for a real world Deep Reinforcement Learning and is built on top of PyTorch.

## Features
+ Composability
  A sampling phase and a learning alghrithm are independent. They comunicate each other only via Policy and Trajectory.

By the composability, we can easily implement following configurations which are difficult for other RL libraries.
+ An agent learns in switched environment (e.g. simulated environment and real world environment).
+ An agent learns by multiple algorithms rather than a single algorithm (e.g. Q-Prop is combination of TRPO and DDPG).
+ Hyper parameters for an algorithm are changing dynamically (e.g. Meta Learning).

To obtain the composability, machina's sampling method is restricted to be episode-based. Episode-based sampling is suitable for real world environment. Some algorithms like updating step by step (e.g. DQN, DDPG) are not reproduced their results exactly with machina.


## install

```
git clone https://github.com/DeepX-inc/machina.git
cd ./machina
pip install -e .
```

## Quick Start
Let's start machina with a quick start Jupyter notebook (some links are to be here).

You can check implemented algorithms in examples (link) for the next step.




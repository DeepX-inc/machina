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
+ An agent learns in switched environment (e.g. simulated environment and real world environment).
+ An agent learns by multiple algorithms rather than a single algorithm (e.g. Q-Prop is combination of TRPO and DDPG).
+ Hyperparameters for an algorithm are changing dynamically (e.g. Meta Learning).

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



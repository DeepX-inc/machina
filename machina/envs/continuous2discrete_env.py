"""
Continuous to discrete.
"""

import gym
import numpy as np


class C2DEnv(object):
    """
    Wrapper environment for converting continuous action space to multi discrete action space.

    Parameters
    ----------
    env : gym.Env
    n_bins : int
        Number of bins for converting continuous to discrete.
        e.g. continuous action space is 0 ~ 1 and n_bins=5,
        action space is converted to [0, 0.25, 0.5, 0.75, 1]
    """

    def __init__(self, env, n_bins=30):
        assert isinstance(env.action_space, gym.spaces.Box)
        assert len(env.action_space.shape) == 1
        self.env = env
        self.n_bins = n_bins
        self.action_space = gym.spaces.MultiDiscrete(
            env.action_space.shape[0] * [n_bins])
        self.observation_space = self.env.observation_space
        if hasattr(env, 'original_env'):
            self.original_env = env.original_env
        else:
            self.original_env = env

    @property
    def horizon(self):
        if hasattr(self.env, 'horizon'):
            return self.env._horizon

    def reset(self):
        return self.env.reset()

    def step(self, action):
        continuous_action = []
        for a, low, high in zip(action, self.env.action_space.low, self.env.action_space.high):
            continuous_action.append(np.linspace(low, high, self.n_bins)[a])
        action = np.array(continuous_action)
        next_obs, reward, done, info = self.env.step(action)
        return next_obs, reward, done, info

    def render(self):
        self.env.render()

    def terminate(self):
        self.env.terminate()

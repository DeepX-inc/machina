import gym
import numpy as np


class RewInObEnv(gym.Env):
    def __init__(self, env, dim=0, normalize=True, initial_value=0, mean=0, std=1, low=-np.inf, high=np.inf):
        self.env = env
        if hasattr(env, 'original_env'):
            self.original_env = env.original_env
        else:
            self.original_env = env
        self.dim = dim
        self.normalize = normalize
        self.initial_value = initial_value
        self.mean = mean
        self.std = std

        observation_space = self.env.observation_space
        action_space = self.env.action_space
        low = np.concatenate(
            [observation_space.low, np.array([low])], axis=dim)
        high = np.concatenate(
            [observation_space.high, np.array([high])], axis=dim)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = self.env.action_space

    @property
    def horizon(self):
        if hasattr(self.env, 'horizon'):
            return self.env._horizon

    def reset(self):
        ob = self.env.reset()
        initial_rew = np.ones((1, )) * self.initial_value
        ob = np.concatenate([ob, initial_rew], axis=self.dim)
        return ob

    def step(self, action):
        next_ob, reward, done, info = self.env.step(action)
        _rew = reward
        if self.normalize:
            _rew = (_rew - self.mean) / self.std
        next_ob = np.concatenate([next_ob, np.array([_rew])], axis=self.dim)
        return next_ob, reward, done, info

    def render(self):
        self.env.render()

    def terminate(self):
        self.env.terminate()

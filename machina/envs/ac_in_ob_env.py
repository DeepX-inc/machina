import gym
import numpy as np


class AcInObEnv(gym.Env):
    def __init__(self, env, dim=0, normalize=True, initial_value=0):
        self.env = env
        if hasattr(env, 'original_env'):
            self.original_env = env.original_env
        else:
            self.original_env = env
        self.dim = dim
        self.normalize = normalize
        self.initial_value = initial_value

        observation_space = self.env.observation_space
        action_space = self.env.action_space
        low = np.concatenate(
            [observation_space.low, action_space.low], axis=dim)
        high = np.concatenate(
            [observation_space.high, action_space.high], axis=dim)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = self.env.action_space

    @property
    def horizon(self):
        if hasattr(self.env, 'horizon'):
            return self.env._horizon

    def reset(self):
        ob = self.env.reset()
        initial_ac = np.ones_like(self.action_space.low) * self.initial_value
        ob = np.concatenate([ob, initial_ac], axis=self.dim)
        return ob

    def step(self, action):
        next_ob, reward, done, info = self.env.step(action)
        if self.normalize:
            lb, ub = self.action_space.low, self.action_space.high
            action = (action - lb) * 2 / (ub - lb) - 1
        next_ob = np.concatenate([next_ob, action], axis=self.dim)
        return next_ob, reward, done, info

    def render(self):
        self.env.render()

    def terminate(self):
        self.env.terminate()

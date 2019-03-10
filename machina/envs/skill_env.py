import gym
import numpy as np


class SkillEnv(gym.Wrapper):
    def __init__(self, env, num_skill=4):
        gym.Wrapper.__init__(self, env)
        self.num_skill = num_skill
        self.skill = 0
        self.real_observation_space = env.observation_space
        low = np.hstack((env.observation_space.low, np.zeros(self.num_skill)))
        high = np.hstack((env.observation_space.high, np.ones(self.num_skill)))
        self.observation_space = gym.spaces.Box(low=low, high=high)
        self.skill_space = gym.spaces.Box(low=np.zeros(
            self.num_skill), high=np.ones(self.num_skill))

    def reset(self, **kwargs):
        # sample skill id
        self.skill = self.unwrapped.np_random.randint(0, self.num_skill)
        obs = self.env.reset(**kwargs)
        obs_skill = np.hstack((obs, np.eye(self.num_skill)[self.skill]))
        return obs_skill

    def step(self, ac):
        obs, reward, done, info = self.env.step(ac)
        obs_skill = np.hstack((obs, np.eye(self.num_skill)[self.skill]))
        return obs_skill, reward, done, info

    @property
    def horizon(self):
        if hasattr(self.env, 'horizon'):
            return self._horizon

    def terminate(self):
        if self.monitoring:
            self.env._close()

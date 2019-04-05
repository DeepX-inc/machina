# The MIT License (MIT)
#
# Copyright (c) 2016 rllab contributors
#
# rllab uses a shared copyright model: each contributor holds copyright over
# their contributions to rllab. The project versioning records all such
# contribution and copyright details.
# By contributing to the rllab repository through pull-request, comment,
# or otherwise, the contributor releases their content to the license and
# copyright terms herein.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""
This is code for gym.
This code is taken from rllab which is MIT-licensed.
"""

import gym
from machina import logger


class CappedCubicVideoSchedule(object):
    # Copied from gym, since this method is frequently moved around
    def __call__(self, count):
        if count < 1000:
            return int(round(count ** (1. / 3))) ** 3 == count
        else:
            return count % 1000 == 0


class NoVideoSchedule(object):
    def __call__(self, count):
        return False


class GymEnv(gym.Env):
    def __init__(self, env, record_video=False, video_schedule=None, log_dir=None,
                 force_reset=False):

        if isinstance(env, str):
            env = gym.envs.make(env)
        self.env = env
        if hasattr(env, 'original_env'):
            self.original_env = env.original_env
        else:
            self.original_env = env
        if self.env.spec is not None:
            self.env_id = env.spec.id
        else:
            self.env_id = None

        if log_dir is None:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(
                self.env, log_dir, video_callable=video_schedule, force=True)
            self.monitoring = True

        self.observation_space = env.observation_space
        logger.log("observation space: {}".format(self.observation_space))
        self.action_space = env.action_space
        logger.log("action space: {}".format(self.action_space))
        if self.env.spec is not None:
            self._horizon = env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']
        else:
            self._horizon = None
        self._log_dir = log_dir
        self._force_reset = force_reset

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        if self._force_reset and self.monitoring:
            from gym.wrappers.monitoring import Monitor
            assert isinstance(self.env, Monitor)
            recorder = self.env.stats_recorder
            if recorder is not None:
                recorder.done = True
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return next_obs, reward, done, info

    def render(self):
        self.env.render()

    def terminate(self):
        if self.monitoring:
            self.env._close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

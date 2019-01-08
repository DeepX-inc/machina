# Copyright 2018 DeepX Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import gym
from machina.misc import logger


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


class GymEnv(object):
    def __init__(self, env_name, record_video=True, video_schedule=None, log_dir=None, record_log=True,
                 force_reset=False):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log(
                    "Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")

        env = gym.envs.make(env_name)
        self.env = env
        self.env_id = env.spec.id

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
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

        self.ob_space = env.observation_space
        logger.log("observation space: {}".format(self.ob_space))
        self.ac_space = env.action_space
        logger.log("action space: {}".format(self.ac_space))
        self._horizon = env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']
        self._log_dir = log_dir
        self._force_reset = force_reset

    @property
    def observation_space(self):
        return self.ob_space

    @property
    def action_space(self):
        return self.ac_space

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
            if self._log_dir is not None:
                print("""
    ***************************

    Training finished! You can upload results to OpenAI Gym by running the following command:

    python scripts/submit_gym.py %s

    ***************************
                """ % self._log_dir)

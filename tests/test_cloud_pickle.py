import unittest

import numpy as np
import cloudpickle

from machina.traj import Traj
from machina.envs import GymEnv, C2DEnv
from machina.samplers import EpiSampler
from machina.pols import RandomPol, GaussianPol, MultiCategoricalPol, CategoricalPol, DeterministicActionNoisePol, MPCPol, ArgmaxQfPol
from machina.vfuncs import DeterministicSAVfunc, DeterministicSVfunc
from machina.utils import make_redis, get_redis

from simple_net import PolNet, VNet, QNet, ModelNet


def rew_func(next_obs, acs, mean_obs=0., std_obs=1., mean_acs=0., std_acs=1.):
    next_obs = next_obs * std_obs + mean_obs
    acs = acs * std_acs + mean_acs
    # Pendulum
    rews = -(torch.acos(next_obs[:, 0].clamp(min=-1, max=1))**2 +
             0.1*(next_obs[:, 2].clamp(min=-8, max=8)**2) + 0.001 * acs.squeeze(-1)**2)
    rews = rews.squeeze(0)

    return rews


class TestCloudPickle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        env = GymEnv('Pendulum-v0')
        random_pol = RandomPol(cls.env.observation_space, cls.env.action_space)
        sampler = EpiSampler(cls.env, pol, num_parallel=1)
        epis = sampler.sample(pol, max_steps=32)
        traj = Traj()
        traj.add_epis(epis)
        traj.register_epis()

        cls.num_step = traj.num_step

        make_redis('localhost', '6379')
        cls.r = get_redis()

        cls.r.set('env', env)
        cls.r.set('traj', traj)

        pol_net = PolNet(env.observation_space, env.action_space)
        gpol = GaussianPol(env.observation_space, env.action_space, pol_net)
        pol_net = PolNet(env.observation_space,
                         env.action_space, deterministic=True)
        dpol = DeterministicActionNoisePol(
            env.observation_space, env.action_space, pol_net)
        model_net = ModelNet(env.observation_space, env.action_space)
        mpcpol = MPCPol(env.observation_space,
                        env.action_space, model_net, rew_func)
        q_net = QNet(env.observation_space, env.action_space)
        qfunc = DeterministicSAVfunc(
            env.observation_space, env.action_space, q_net)
        aqpol = ArgmaxQfPol(env.observation_space, env.action_space, qfunc)
        v_net = VNet(env.observation_space)
        vfunc = DeterministicSVfunc(env.observation_space, v_net)

        cls.r.set('gpol', cloudpickle.dumps(gpol))
        cls.r.set('dpol', cloudpickle.dumps(dpol))
        cls.r.set('mpcpol', cloudpickle.dumps(mpcpol))
        cls.r.set('qfunc', cloudpickle.dumps(qfunc))
        cls.r.set('aqpol', cloudpickle.dumps(aqpol))
        cls.r.set('vfunc', cloudpickle.dumps(vfunc))

        c2d = C2DEnv(env)
        pol_net = PolNet(c2d.observation_space, c2d.action_space)
        mcpol = MultiCategoricalPol(
            env.observation_space, env.action_space, pol_net)

        cls.r.set('mcpol', cloudpickle.dumps(mcpol))

    @classmethod
    def tearDownClass(cls):
        cloudpickle.loads(cls.r.get('env'))
        cloudpickle.loads(cls.r.get('traj'))
        cloudpickle.loads(cls.r.get('gpol'))
        cloudpickle.loads(cls.r.get('dpol'))
        cloudpickle.loads(cls.r.get('mpcpol'))
        cloudpickle.loads(cls.r.get('qfunc'))
        cloudpickle.loads(cls.r.get('aqpol'))
        cloudpickle.loads(cls.r.get('vfunc'))
        cloudpickle.loads(cls.r.get('mcpol'))


if __name__ == '__main__':
    unittest.main()

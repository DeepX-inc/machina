"""
Sampler using ray
"""

import copy
import time

import ray
import gym
import numpy as np
import torch
import torch.multiprocessing as mp

from machina.utils import cpu_mode


LARGE_NUMBER = 100000000


class Worker(object):
    def __init__(self, pol, env, seed, worker_id, prepro=None):
        self.set_pol(pol)
        self.env = env
        self.worker_id = worker_id
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.set_num_threads(1)
        if prepro is None:
            self.prepro = lambda x: x
        else:
            self.prepro = prepro

    def set_pol(self, pol):
        self.pol = pol
        self.pol.eval()
        self.pol.dp_run = False

    def set_pol_state(self, state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.find("dp_net") == -1:
                new_state_dict[k] = v
        self.pol.load_state_dict(new_state_dict)
        self.pol.eval()

    @classmethod
    def as_remote(cls, resources={}):
        return ray.remote(resources=resources)(cls)

    def one_epi(self, deterministic=False):
        with cpu_mode():
            obs = []
            acs = []
            rews = []
            dones = []
            a_is = []
            e_is = []
            o = self.env.reset()
            self.pol.reset()
            done = False
            epi_length = 0
            while not done:
                o = self.prepro(o)
                if not deterministic:
                    ac_real, ac, a_i = self.pol(
                        torch.tensor(o, dtype=torch.float))
                else:
                    ac_real, ac, a_i = self.pol.deterministic_ac_real(
                        torch.tensor(o, dtype=torch.float))
                ac_real = ac_real.reshape(self.pol.action_space.shape)
                next_o, r, done, e_i = self.env.step(np.array(ac_real))
                obs.append(o)
                rews.append(r)
                dones.append(done)
                acs.append(ac.squeeze().detach().cpu(
                ).numpy().reshape(self.pol.action_space.shape))
                _a_i = dict()
                for key in a_i.keys():
                    if a_i[key] is None:
                        continue
                    if isinstance(a_i[key], tuple):
                        _a_i[key] = tuple([h.squeeze().detach().cpu().numpy()
                                           for h in a_i[key]])
                    else:
                        _a_i[key] = a_i[key].squeeze().detach(
                        ).cpu().numpy().reshape(self.pol.a_i_shape)
                a_i = _a_i
                a_is.append(a_i)
                e_is.append(e_i)
                epi_length += 1
                if done:
                    break
                o = next_o

            return epi_length, dict(
                obs=np.array(obs, dtype='float32'),
                acs=np.array(acs, dtype='float32'),
                rews=np.array(rews, dtype='float32'),
                dones=np.array(dones, dtype='float32'),
                a_is=dict([(key, np.array([a_i[key] for a_i in a_is], dtype='float32'))
                           for key in a_is[0].keys()]),
                e_is=dict([(key, np.array([e_i[key] for e_i in e_is], dtype='float32'))
                           for key in e_is[0].keys()])
            )


class EpiSampler(object):
    """
    A sampler which sample episodes.

    Parameters
    ----------
    env : gym.Env
    pol : Pol
    num_parallel : int
        Number of processes
    prepro : Prepro
    seed : int
    node_info: dict
        This is used to control worker scheduling using ray custom resources.
        (See https://ray.readthedocs.io/en/latest/resources.html#custom-resources).
        To use this feature, ray nodes must start with some custom resources.
        For example, start ray `ray start --head --resources='{"node1": 100}' on
        node 1 and `ray start --redis-address <addr> --resources='{"node2": 100}'`
        on node 2. If node_info is {"node1": 10, "node2": 10}, then
        10 workers require resources "node1" and 10 workers requires resource "node2".
        As a result, 10 workers scheduled on node 1 and 10 workers on node 2.
        Default (empty node_info) is using ray scheduling policy.
    """

    def __init__(self, pol, env, num_parallel=8, prepro=None, seed=256,
                 node_info={}):
        pol = copy.deepcopy(pol)
        pol.to('cpu')

        pol = ray.put(pol)
        env = ray.put(env)

        resources = []
        for k, v in node_info.items():
            for _ in range(v):
                resources.append({k: 1})
        assert len(resources) <= num_parallel
        if len(resources) < num_parallel:
            for _ in range(num_parallel - len(resources)):
                resources.append({})

        self.workers = [Worker.as_remote(resources=r).remote(pol, env, seed, i, prepro)
                        for i, r in zip(range(num_parallel), resources)]

    def set_pol(self, pol):
        if not isinstance(pol, ray.ObjectID):
            pol = ray.put(pol)
        for w in self.workers:
            w.set_pol.remote(pol)

    def set_pol_state(self, state_dict):
        if not isinstance(state_dict, ray.ObjectID):
            state_dict = ray.put(state_dict)
        for w in self.workers:
            w.set_pol_state.remote(state_dict)

    def sample(self, max_epis=None, max_steps=None, deterministic=False):
        """
        Switch on sampling processes.

        Parameters
        ----------
        pol : Pol
        max_epis : int or None
            maximum episodes of episodes.
            If None, this value is ignored.
        max_steps : int or None
            maximum steps of episodes
            If None, this value is ignored.
        deterministic : bool

        Returns
        -------
        epis : list of dict
            Sampled epis.

        Raises
        ------
        ValueError
            If max_steps and max_epis are botch None.
        """

        if max_epis is None and max_steps is None:
            raise ValueError(
                'Either max_epis or max_steps needs not to be None')
        max_epis = max_epis if max_epis is not None else LARGE_NUMBER
        max_steps = max_steps if max_steps is not None else LARGE_NUMBER

        epis = []
        n_steps = 0
        n_epis = 0

        pending = {w.one_epi.remote(deterministic): w for w in self.workers}

        while pending:
            ready, _ = ray.wait(list(pending))
            for obj_id in ready:
                worker = pending.pop(obj_id)
                (l, epi) = ray.get(obj_id)
                epis.append(epi)
                n_steps += l
                n_epis += 1
                if n_steps < max_steps and (n_epis + len(pending)) < max_epis:
                    pending[worker.one_epi.remote(deterministic)] = worker

        return epis

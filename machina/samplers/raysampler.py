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


@ray.remote(num_cpus=1)
class Worker(object):
    def __init__(self, env, seed, worker_id, prepro=None):
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

            return self.worker_id, epi_length, dict(
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
    """

    def __init__(self, env, num_parallel=8, prepro=None, seed=256):
        self.env = env
        self.num_parallel = num_parallel

        self.n_steps_global = 0
        self.max_steps = 0
        self.n_epis_global = 0
        self.max_epis = 0

        env = ray.put(self.env)

        self.workers = [Worker.remote(env, seed, i, prepro)
                        for i in range(num_parallel)]

    def sample(self, pol, max_epis=None, max_steps=None, deterministic=False):
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

        pol = copy.deepcopy(pol)
        pol.eval()
        pol = ray.put(pol)
        for w in self.workers:
            w.set_pol.remote(pol)

        if max_epis is None and max_steps is None:
            raise ValueError(
                'Either max_epis or max_steps needs not to be None')
        max_epis = max_epis if max_epis is not None else LARGE_NUMBER
        max_steps = max_steps if max_steps is not None else LARGE_NUMBER

        epis = []
        n_steps_global = 0
        n_epis_global = 0

        worker_ids = [i for i in range(len(self.workers))]
        epi_remain = []

        while max_steps > n_steps_global and max_epis > n_epis_global:
            epi_remain += [self.workers[i].one_epi.remote(
                deterministic) for i in worker_ids]
            worker_ids = []

            epi_done, epi_remain = ray.wait(epi_remain)
            for (worker_id, l, epi) in ray.get(epi_done):
                n_steps_global += l
                n_epis_global += 1
                epis.append(epi)
                worker_ids.append(worker_id)

        epi_done = ray.get(epi_remain)
        for (_, l, epi) in epi_done:
            n_steps_global += l
            n_epis_global += 1
            epis.append(epi)

        return epis

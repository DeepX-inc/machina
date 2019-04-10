"""
Sampler class
"""

import copy
import time

import gym
import numpy as np
import torch
import torch.multiprocessing as mp

from machina.utils import cpu_mode


LARGE_NUMBER = 100000000


def one_epi(env, pol, deterministic=False, prepro=None):
    """
    Sampling an episode.

    Parameters
    ----------
    env : gym.Env
    pol : Pol
    deterministic : bool
        If True, policy is deterministic.
    prepro : Prepro

    Returns
    -------
    epi_length, epi : int, dict
    """
    with cpu_mode():
        if prepro is None:
            def prepro(x): return x
        obs = []
        acs = []
        rews = []
        dones = []
        a_is = []
        e_is = []
        o = env.reset()
        pol.reset()
        done = False
        epi_length = 0
        while not done:
            o = prepro(o)
            if not deterministic:
                ac_real, ac, a_i = pol(torch.tensor(o, dtype=torch.float))
            else:
                ac_real, ac, a_i = pol.deterministic_ac_real(
                    torch.tensor(o, dtype=torch.float))
            ac_real = ac_real.reshape(pol.action_space.shape)
            next_o, r, done, e_i = env.step(np.array(ac_real))
            obs.append(o)
            rews.append(r)
            dones.append(done)
            acs.append(ac.squeeze().detach().cpu(
            ).numpy().reshape(pol.action_space.shape))
            _a_i = dict()
            for key in a_i.keys():
                if a_i[key] is None:
                    continue
                if isinstance(a_i[key], tuple):
                    _a_i[key] = tuple([h.squeeze().detach().cpu().numpy()
                                       for h in a_i[key]])
                else:
                    _a_i[key] = a_i[key].squeeze().detach(
                    ).cpu().numpy().reshape(pol.a_i_shape)
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


def mp_sample(pol, env, max_steps, max_epis, n_steps_global, n_epis_global, epis, exec_flag, deterministic_flag, process_id, prepro=None, seed=256):
    """
    Multiprocess sample.
    Sampling episodes until max_steps or max_epis is achieved.

    Parameters
    ----------
    pol : Pol
    env : gym.Env
    max_steps : int
        maximum steps of episodes
    max_epis : int
        maximum episodes of episodes
    n_steps_global : torch.Tensor
        shared Tensor
    n_epis_global : torch.Tensor
        shared Tensor
    epis : list
        multiprocessing's list for sharing episodes between processes.
    exec_flag : torch.Tensor
        execution flag
    deterministic_flag : torch.Tensor
    process_id : int
    prepro : Prepro
    seed : int
    """

    np.random.seed(seed + process_id)
    torch.manual_seed(seed + process_id)
    torch.set_num_threads(1)

    while True:
        time.sleep(0.1)
        if exec_flag > 0:
            while max_steps > n_steps_global and max_epis > n_epis_global:
                l, epi = one_epi(env, pol, deterministic_flag, prepro)
                n_steps_global += l
                n_epis_global += 1
                epis.append(epi)
            exec_flag.zero_()


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

    def __init__(self, env, pol, num_parallel=8, prepro=None, seed=256):
        self.env = env
        self.pol = copy.deepcopy(pol)
        self.pol.to('cpu')
        self.pol.share_memory()
        self.pol.eval()
        self.num_parallel = num_parallel

        self.n_steps_global = torch.tensor(0, dtype=torch.long).share_memory_()
        self.max_steps = torch.tensor(0, dtype=torch.long).share_memory_()
        self.n_epis_global = torch.tensor(
            0, dtype=torch.long).share_memory_()
        self.max_epis = torch.tensor(0, dtype=torch.long).share_memory_()

        self.exec_flags = [torch.tensor(
            0, dtype=torch.long).share_memory_() for _ in range(self.num_parallel)]
        self.deterministic_flag = torch.tensor(
            0, dtype=torch.uint8).share_memory_()

        self.epis = mp.Manager().list()
        self.processes = []
        for ind in range(self.num_parallel):
            p = mp.Process(target=mp_sample, args=(self.pol, env, self.max_steps, self.max_epis, self.n_steps_global,
                                                   self.n_epis_global, self.epis, self.exec_flags[ind], self.deterministic_flag, ind, prepro, seed))
            p.start()
            self.processes.append(p)

    def __del__(self):
        for p in self.processes:
            p.terminate()

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
        for sp, p in zip(self.pol.parameters(), pol.parameters()):
            sp.data.copy_(p.data.to('cpu'))

        if max_epis is None and max_steps is None:
            raise ValueError(
                'Either max_epis or max_steps needs not to be None')
        max_epis = max_epis if max_epis is not None else LARGE_NUMBER
        max_steps = max_steps if max_steps is not None else LARGE_NUMBER

        self.n_steps_global.zero_()
        self.n_epis_global.zero_()

        self.max_steps.zero_()
        self.max_steps += max_steps
        self.max_epis.zero_()
        self.max_epis += max_epis

        if deterministic:
            self.deterministic_flag.zero_()
            self.deterministic_flag += 1
        else:
            self.deterministic_flag.zero_()

        del self.epis[:]

        for exec_flag in self.exec_flags:
            exec_flag += 1

        while True:
            if all([exec_flag == 0 for exec_flag in self.exec_flags]):
                return list(self.epis)

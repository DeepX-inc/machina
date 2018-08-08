import copy

import numpy as np
import torch
import torch.multiprocessing as mp

from machina.utils import cpu_mode
from machina.samplers.base import BaseSampler

def one_path(env, pol, prepro=None):
    if prepro is None:
        prepro = lambda x: x
    obs = []
    acs = []
    rews = []
    a_is = []
    e_is = []
    o = env.reset()
    d = False
    path_length = 0
    while not d:
        o = prepro(o)
        ac_real, ac, a_i = pol(torch.tensor(o, dtype=torch.float).unsqueeze(0))
        next_o, r, d, e_i = env.step(ac_real[0])
        obs.append(o)
        rews.append(r)
        acs.append(ac.detach().numpy()[0])
        a_i = dict([(key, a_i[key].detach().numpy()[0]) for key in a_i.keys()])
        a_is.append(a_i)
        e_is.append(e_i)
        path_length += 1
        if d:
            break
        o = next_o
    return path_length, dict(
        obs=np.array(obs, dtype='float32'),
        acs=np.array(acs, dtype='float32'),
        rews=np.array(rews, dtype='float32'),
        a_is=dict([(key, np.array([a_i[key] for a_i in a_is], dtype='float32')) for key in a_is[0].keys()]),
        e_is=dict([(key, np.array([e_i[key] for e_i in e_is], dtype='float32')) for key in e_is[0].keys()])
    )

def many_paths(env, pol, max_samples, max_episodes, n_samples_global, n_episodes_global, paths, prepro=None):
    while max_samples > n_samples_global and max_episodes > n_episodes_global:
        l, path = one_path(env, pol, prepro)
        n_samples_global += l
        n_episodes_global += 1
        paths.append(path)

class ParallelSampler(BaseSampler):
    def __init__(self, env, num_parallel=8):
        BaseSampler.__init__(self, env)
        self.num_parallel = num_parallel

    def sample(self, pol, max_samples, max_episodes, prepro=None):
        sampling_pol = copy.deepcopy(pol)
        sampling_pol = sampling_pol.cpu()
        n_samples_global = torch.tensor(0, dtype=torch.long).share_memory_()
        n_episodes_global = torch.tensor(0, dtype=torch.long).share_memory_()
        paths = mp.Manager().list()
        with cpu_mode():
            processes = []
            for _ in range(self.num_parallel):
                p = mp.Process(target=many_paths, args=(self.env, pol, max_samples, max_episodes, n_samples_global, n_episodes_global, paths, prepro))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        return list(paths)


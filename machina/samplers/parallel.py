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
        acs.append(ac.detach().cpu().numpy()[0])
        a_i = dict([(key, a_i[key].detach().cpu().numpy()[0]) for key in a_i.keys()])
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

def sample_process(pol, env, max_samples, max_episodes, n_samples_global, n_episodes_global, paths, exec_flags, process_id, prepro=None):
    while True:
        if exec_flags[process_id] > 0:
            while max_samples > n_samples_global and max_episodes > n_episodes_global:
                l, path = one_path(env, pol, prepro)
                n_samples_global += l
                n_episodes_global += 1
                paths.append(path)
            exec_flags[process_id].zero_()

class ParallelSampler(BaseSampler):
    def __init__(self, env, pol, max_samples, max_episodes, num_parallel=8, prepro=None):
        BaseSampler.__init__(self, env)
        self.pol = copy.deepcopy(pol.cpu())
        self.pol.share_memory()
        self.max_samples = max_samples
        self.max_episodes = max_episodes
        self.num_parallel = num_parallel

        self.n_samples_global = torch.tensor(0, dtype=torch.long).share_memory_()
        self.n_episodes_global = torch.tensor(0, dtype=torch.long).share_memory_()
        self.exec_flags = [torch.tensor(0, dtype=torch.long).share_memory_() for _ in range(self.num_parallel)]

        self.paths = mp.Manager().list()
        self.processes = []
        for ind in range(self.num_parallel):
            p = mp.Process(target=sample_process, args=(pol, env, max_samples, max_episodes, self.n_samples_global, self.n_episodes_global, self.paths, self.exec_flags, ind, prepro))
            p.start()
            self.processes.append(p)

    def __del__(self):
        for p in self.processes:
            p.join()

    def sample(self, pol, *args):
        self.pol.load_state_dict(pol.cpu().state_dict())
        self.n_samples_global.zero_()
        self.n_episodes_global.zero_()
        del self.paths[:]

        for exec_flag in self.exec_flags:
            exec_flag += 1

        while True:
            if all([exec_flag == 0 for exec_flag in self.exec_flags]):
                return list(self.paths)


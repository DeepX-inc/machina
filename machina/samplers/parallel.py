import copy

import numpy as np
import torch
import torch.multiprocessing as mp
import gym

from machina.utils import cpu_mode
from machina.samplers.base import BaseSampler

def one_path(env, pol, deterministic=False, prepro=None):
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
    if pol.rnn:
        hs = pol.init_hs(batch_size=1)
    else:
        hs = None
    while not d:
        o = prepro(o)
        if pol.rnn:
            if not deterministic:
                ac_real, ac, a_i = pol(torch.tensor(o, dtype=torch.float).unsqueeze(0).unsqueeze(0), hs)
            else:
                ac_real, ac, a_i = pol.deterministic_ac_real(torch.tensor(o, dtype=torch.float).unsqueeze(0).unsqueeze(0), hs)
            #TODO: fix for multi discrete
            if isinstance(pol.ac_space, gym.spaces.Box):
                ac_real = ac_real.reshape(*pol.ac_space.shape)
            else:
                ac_real = ac_real.reshape(())
            hs = a_i['hs']
        else:
            if not deterministic:
                ac_real, ac, a_i = pol(torch.tensor(o, dtype=torch.float).unsqueeze(0))
            else:
                ac_real, ac, a_i = pol.deterministic_ac_real(torch.tensor(o, dtype=torch.float).unsqueeze(0))
            #TODO: fix for multi discrete
            if isinstance(pol.ac_space, gym.spaces.Box):
                ac_real = ac_real.reshape(*pol.ac_space.shape)
            else:
                ac_real = ac_real.reshape(())
        next_o, r, d, e_i = env.step(np.array(ac_real))
        obs.append(o)
        rews.append(r)
        #TODO: fix for multi discrete
        if isinstance(pol.ac_space, gym.spaces.Box):
            acs.append(ac.squeeze().detach().cpu().numpy().reshape(*pol.ac_space.shape))
        else:
            acs.append(ac.squeeze().detach().cpu().numpy().reshape(()))
        _a_i = dict()
        for key in a_i.keys():
            if a_i[key] is None:
                continue
            if isinstance(a_i[key], tuple):
                _a_i[key] = tuple([h.squeeze().detach().cpu().numpy() for h in a_i[key]])
            else:
                #TODO: fix for multi discrete
                if isinstance(pol.ac_space, gym.spaces.Box):
                    _a_i[key] = a_i[key].squeeze().detach().cpu().numpy().reshape(*pol.ac_space.shape)
                else:
                    _a_i[key] = a_i[key].squeeze().detach().cpu().numpy().reshape((pol.ac_space.n, ))
        a_i = _a_i
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

def sample_process(pol, env, max_samples, max_episodes, n_samples_global, n_episodes_global, paths, exec_flags, deterministic_flag, process_id, prepro=None, seed=256):

    np.random.seed(seed + process_id)
    torch.manual_seed(seed + process_id)
    torch.set_num_threads(1)

    while True:
        if exec_flags[process_id] > 0:
            while max_samples > n_samples_global and max_episodes > n_episodes_global:
                l, path = one_path(env, pol, deterministic_flag, prepro)
                n_samples_global += l
                n_episodes_global += 1
                paths.append(path)
            exec_flags[process_id].zero_()

class ParallelSampler(BaseSampler):
    def __init__(self, env, pol, max_samples, max_episodes, num_parallel=8, prepro=None, seed=256):
        BaseSampler.__init__(self, env)
        self.pol = copy.deepcopy(pol)
        self.pol.to('cpu')
        self.pol.share_memory()
        self.pol.eval()
        self.max_samples = max_samples
        self.max_episodes = max_episodes
        self.num_parallel = num_parallel

        self.n_samples_global = torch.tensor(0, dtype=torch.long).share_memory_()
        self.n_episodes_global = torch.tensor(0, dtype=torch.long).share_memory_()
        self.exec_flags = [torch.tensor(0, dtype=torch.long).share_memory_() for _ in range(self.num_parallel)]
        self.deterministic_flag = torch.tensor(0, dtype=torch.uint8).share_memory_()

        self.paths = mp.Manager().list()
        self.processes = []
        for ind in range(self.num_parallel):
            p = mp.Process(target=sample_process, args=(self.pol, env, max_samples, max_episodes, self.n_samples_global, self.n_episodes_global, self.paths, self.exec_flags, self.deterministic_flag, ind, prepro, seed))
            p.start()
            self.processes.append(p)

    def __del__(self):
        for p in self.processes:
            p.join()

    def sample(self, pol, *args, **kwargs):
        deterministic = kwargs.pop('deterministic', False)
        if deterministic:
            self.deterministic_flag.zero_()
            self.deterministic_flag += 1
        else:
            self.deterministic_flag.zero_()

        for sp, p in zip(self.pol.parameters(), pol.parameters()):
            sp.data.copy_(p.data.to('cpu'))

        self.n_samples_global.zero_()
        self.n_episodes_global.zero_()

        del self.paths[:]

        for exec_flag in self.exec_flags:
            exec_flag += 1

        while True:
            if all([exec_flag == 0 for exec_flag in self.exec_flags]):
                return list(self.paths)


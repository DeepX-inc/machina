import copy
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import gym

from machina.utils import cpu_mode
from machina.misc import logger
from machina.samplers.base import BaseSampler


def sample_process(pol, env, max_samples, paths, exec_flags, process_id, prepro=None, seed=256):
    if prepro is None:
        prepro = lambda x: x

    np.random.seed(seed + process_id)
    torch.manual_seed(seed + process_id)
    torch.set_num_threads(1)

    d = False
    o = env.reset()
    if pol.rnn:
        hs = pol.init_hs(batch_size=1)
    else:
        hs = None
    while True:
        if exec_flags[process_id] > 0:
            n_samples = 0
            obs = []
            acs = []
            rews = []
            dones = []
            a_is = []
            e_is = []
            init_hs = hs
            if pol.rnn:
                hs = (hs[0].detach(), hs[1].detach())
            while max_samples > n_samples:
                if d:
                    o = env.reset()
                if pol.rnn:
                    ac_real, ac, a_i = pol(torch.tensor(o, dtype=torch.float).unsqueeze(0).unsqueeze(0), hs)
                    if isinstance(pol.ac_space, gym.spaces.Box):
                        ac_real = ac_real.reshape(*pol.ac_space.shape)
                    else:
                        ac_real = ac_real.reshape(())
                    hs = a_i['hs']
                else:
                    ac_real, ac, a_i = pol(torch.tensor(o, dtype=torch.float).unsqueeze(0))
                next_o, r, next_d, e_i = env.step(np.array(ac_real))
                obs.append(o)
                rews.append(r)
                if isinstance(pol.ac_space, gym.spaces.Box):
                    acs.append(ac.squeeze().detach().cpu().numpy().reshape(*pol.ac_space.shape))
                else:
                    acs.append(ac.squeeze().detach().cpu().numpy().reshape(()))
                dones.append(d)
                _a_i = dict()
                for key in a_i.keys():
                    if a_i[key] is None:
                        continue
                    if isinstance(a_i[key], tuple):
                        _a_i[key] = tuple([h.squeeze().detach().cpu().numpy() for h in a_i[key]])
                    else:
                        if isinstance(pol.ac_space, gym.spaces.Box):
                            _a_i[key] = a_i[key].squeeze().detach().cpu().numpy().reshape(*pol.ac_space.shape)
                        else:
                            _a_i[key] = a_i[key].squeeze().detach().cpu().numpy().reshape((pol.ac_space.n, ))
                a_i = _a_i
                a_is.append(a_i)
                e_is.append(e_i)
                o = next_o
                d = next_d
                n_samples += 1
            last_ob = o
            last_d = d
            path = dict(
                obs=np.array(obs, dtype='float32'),
                acs=np.array(acs, dtype='float32'),
                rews=np.array(rews, dtype='float32'),
                dones=np.array(dones, dtype='float32'),
                last_ob=np.array(last_ob, dtype='float32'),
                last_d=np.array(last_d, dtype='float32'),
                init_hs=tuple([h.detach().cpu().numpy() for h in init_hs]),
                last_hs=tuple([h.detach().cpu().numpy() for h in hs]),
                a_is=dict([(key, np.array([a_i[key] for a_i in a_is], dtype='float32')) for key in a_is[0].keys()]),
                e_is=dict([(key, np.array([e_i[key] for e_i in e_is], dtype='float32')) for key in e_is[0].keys()])
            )
            paths.append(path)
            exec_flags[process_id].zero_()

class ParallelVectorSampler(BaseSampler):
    def __init__(self, env, pol, max_samples, num_parallel=8, prepro=None, seed=256):
        super(ParallelVectorSampler, self).__init__(env)
        self.pol = copy.deepcopy(pol)
        self.pol.to('cpu')
        self.pol.share_memory()
        self.pol.eval()
        self.max_samples = max_samples
        self.num_parallel = num_parallel

        self.exec_flags = [torch.tensor(0, dtype=torch.long).share_memory_() for _ in range(self.num_parallel)]

        self.paths = mp.Manager().list()
        self.processes = []
        for ind in range(self.num_parallel):
            p = mp.Process(target=sample_process, args=(self.pol, env, max_samples, self.paths, self.exec_flags, ind, prepro, seed))
            p.start()
            self.processes.append(p)

    def __del__(self):
        for p in self.processes:
            p.join()

    def sample(self, pol, too_long_to_wait_time=500, *args, **kwargs):
        for sp, p in zip(self.pol.parameters(), pol.parameters()):
            sp.data.copy_(p.data.to('cpu'))

        del self.paths[:]

        for exec_flag in self.exec_flags:
            exec_flag += 1

        while True:
            if all([exec_flag == 0 for exec_flag in self.exec_flags]):
                return list(self.paths)


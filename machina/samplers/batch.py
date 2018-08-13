import copy
import numpy as np
import torch
from machina.utils import cpu_mode
from machina.samplers.base import BaseSampler


class BatchSampler(BaseSampler):
    def __init__(self, env):
        BaseSampler.__init__(self, env)

    def one_path(self, pol, prepro=None):
        if prepro is None:
            prepro = lambda x: x
        obs = []
        acs = []
        rews = []
        a_is = []
        e_is = []
        o = self.env.reset()
        pol.reset()
        d = False
        path_length = 0
        while not d:
            o = prepro(o)
            ac_real, ac, a_i = pol(torch.tensor(o, dtype=torch.float).unsqueeze(0))
            next_o, r, d, e_i = self.env.step(ac_real[0])
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

    def sample(self, pol, max_samples, max_episodes, prepro=None):
        sampling_pol = copy.deepcopy(pol)
        sampling_pol = sampling_pol.cpu()
        n_samples = 0
        n_episodes = 0
        paths = []
        with cpu_mode():
            while max_samples > n_samples and max_episodes > n_episodes:
                l, path = self.one_path(sampling_pol, prepro)
                n_samples += l
                n_episodes += 1
                paths.append(path)
        return paths


class InvariantBatchSampler(BaseSampler):
    def __init__(self, env, expert_obs, agent_encoder, expert_encoder):
        BaseSampler.__init__(self, env)
        self.expert_obs = expert_obs
        self.agent_encoder = agent_encoder
        self.expert_encoder = expert_encoder

    def one_path(self, pol, n_episodes, prepro=None):
        if prepro is None:
            prepro = lambda x: x
        obs = []
        acs = []
        rews = []
        a_is = []
        e_is = []
        n_episodes = n_episodes+100
        self.env.seed(n_episodes+1)
        o = self.env.reset()
        pol.reset()
        d = False
        path_length = 0
        while not d:
            o = prepro(o)
            ac_real, ac, a_i = pol(Variable(torch.from_numpy(o).float().unsqueeze(0)))
            next_o, r, d, e_i = self.env.step(ac_real[0])
            obs.append(o)
            rews.append(r)
            acs.append(ac.data.cpu().numpy()[0])
            a_i = dict([(key, a_i[key].data.cpu().numpy()[0]) for key in a_i.keys()])
            a_is.append(a_i)
            e_is.append(e_i)
            path_length += 1
            if d:
                break
            o = next_o
        obs_array = np.array(obs, dtype='float32')
        obs_array = np.delete(np.delete(obs_array, axis=1, obj=6), axis=1, obj=6)
        expert_obs_array = self.expert_obs[n_episodes]
        self.agent_encoder.eval()
        self.expert_encoder.eval()
        z_agent = self.agent_encoder(Variable(np2torch(obs_array).float(), volatile=True)).data.cpu().numpy()
        z_expert = self.expert_encoder(Variable(np2torch(expert_obs_array).float(), volatile=True)).data.cpu().numpy()
        pseudo_rews = (z_agent-z_expert)**2

        return path_length, dict(
            obs=np.array(obs, dtype='float32'),
            acs=np.array(acs, dtype='float32'),
            rews=pseudo_rews,
            real_rews=np.array(rews, dtype='float32'),
            a_is=dict([(key, np.array([a_i[key] for a_i in a_is], dtype='float32')) for key in a_is[0].keys()]),
            e_is=dict([(key, np.array([e_i[key] for e_i in e_is], dtype='float32')) for key in e_is[0].keys()])
        )

    def sample(self, pol, max_samples, max_episodes, prepro=None):
        sampling_pol = copy.deepcopy(pol)
        sampling_pol = sampling_pol.cpu()
        n_samples = 0
        n_episodes = 0
        paths = []
        with cpu_mode():
            while max_samples > n_samples and max_episodes > n_episodes:
                l, path = self.one_path(sampling_pol, n_episodes, prepro)
                n_samples += l
                n_episodes += 1
                paths.append(path)
        return paths

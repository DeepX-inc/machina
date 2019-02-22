"""
These are functions which is applied to episodes.
"""

import numpy as np
import copy
import torch
import torch.nn.functional as F

from machina.utils import get_device
from machina import loss_functional as lf


def compute_vs(data, vf):
    """
    Computing Value Function.

    Parameters
    ----------
    data : Traj
    vf : SVFunction

    Returns
    -------
    data : Traj
    """
    epis = data.current_epis
    vf.reset()
    with torch.no_grad():
        for epi in epis:
            if vf.rnn:
                obs = torch.tensor(
                    epi['obs'], dtype=torch.float, device=get_device()).unsqueeze(1)
            else:
                obs = torch.tensor(
                    epi['obs'], dtype=torch.float, device=get_device())
            epi['vs'] = vf(obs)[0].detach().cpu().numpy()

    return data


def set_all_pris(data, pri):
    epis = data.current_epis
    for epi in epis:
        pris = pri.repeat(len(epi['obs']))
        epi['pris'] = pris.cpu().numpy()
    return data


def compute_pris(data, qf, targ_qf, targ_pol, gamma, continuous=True, deterministic=True, sampling=1, alpha=0.6, epsilon=1e-6):
    if continuous:
        epis = data.current_epis
        for epi in epis:
            data_map = dict()
            keys = ['obs', 'acs', 'rews', 'next_obs', 'dones']
            for key in keys:
                data_map[key] = torch.tensor(epi[key], device=get_device())
            with torch.no_grad():
                bellman_loss = lf.bellman(
                    qf, targ_qf, targ_pol, data_map, gamma, continuous, deterministic, sampling, reduction='none')
                td_loss = torch.sqrt(bellman_loss*2)
                pris = (torch.abs(td_loss) + epsilon) ** alpha
                epi['pris'] = pris.cpu().numpy()
        return data
    else:
        raise NotImplementedError(
            "Only Q function with continuous action space is supported now.")


def compute_rets(data, gamma):
    """
    Computing discounted cumulative returns.

    Parameters
    ----------
    data : Traj
    gamma : float
        Discount rate

    Returns
    -------
    data : Traj
    """
    epis = data.current_epis
    for epi in epis:
        rews = epi['rews']
        rets = np.empty(len(rews), dtype=np.float32)
        last_rew = 0
        for t in reversed(range(len(rews))):
            rets[t] = last_rew = rews[t] + gamma * last_rew
        epi['rets'] = rets

    return data


def compute_advs(data, gamma, lam):
    """
    Computing Advantage Function.

    Parameters
    ----------
    data : Traj
    gamma : float
        Discount rate
    lam : float
        Bias-Variance trade-off parameter

    Returns
    -------
    data : Traj
    """
    epis = data.current_epis
    for epi in epis:
        rews = epi['rews']
        vs = epi['vs']
        vs = np.append(vs, 0)
        advs = np.empty(len(rews), dtype=np.float32)
        last_gaelam = 0
        for t in reversed(range(len(rews))):
            delta = rews[t] + gamma * vs[t + 1] - vs[t]
            advs[t] = last_gaelam = delta + gamma * lam * last_gaelam
        epi['advs'] = advs

    return data


def centerize_advs(data, eps=1e-6):
    """
    Centerizing Advantage Function.

    Parameters
    ----------
    data : Traj
    eps : float
        Small value for preventing 0 division.

    Returns
    -------
    data : Traj
    """
    epis = data.current_epis
    _advs = np.concatenate([epi['advs'] for epi in epis])
    for epi in epis:
        epi['advs'] = (epi['advs'] - np.mean(_advs)) / (np.std(_advs) + eps)

    return data


def add_next_obs(data):
    """
    Adding next observations to episodes.

    Parameters
    ----------
    data : Traj

    Returns
    -------
    data : Traj
    """
    epis = data.current_epis
    for epi in epis:
        obs = epi['obs']
        _obs = [ob for ob in obs]
        next_obs = np.array(_obs[1:] + _obs[:1], dtype=np.float32)
        epi['next_obs'] = next_obs

    return data


def compute_h_masks(data):
    """
    Computing masks for hidden state.
    At the begining of an episode, it remarks 1.

    Parameters
    ----------
    data : Traj

    Returns
    -------
    data : Traj
    """
    epis = data.current_epis
    for epi in epis:
        h_masks = np.zeros_like(epi['rews'])
        h_masks[0] = 1
        epi['h_masks'] = h_masks

    return data


def compute_pseudo_rews(data, rew_giver, state_only=False):
    epis = data.current_epis
    for epi in epis:
        obs = torch.tensor(epi['obs'], dtype=torch.float, device=get_device())
        if state_only:
            logits, _ = rew_giver(obs)
        else:
            acs = torch.tensor(
                epi['acs'], dtype=torch.float, device=get_device())
            logits, _ = rew_giver(obs, acs)
        with torch.no_grad():
            rews = -F.logsigmoid(-logits).cpu().numpy()
        epi['real_rews'] = copy.deepcopy(epi['rews'])
        epi['rews'] = rews
    return data


def train_test_split(epis, train_size):
    num_epi = len(epis)
    num_train = int(num_epi * train_size)
    indices = np.arange(num_epi)
    train_epis, test_epis = [[epis[indice] for indice in indices] for indices in
                             np.array_split(indices, [num_train])]

    return train_epis, test_epis

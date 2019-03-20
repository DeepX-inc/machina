"""
These are functions which is applied to episodes.
"""

import numpy as np
import copy
import torch
import torch.nn.functional as F

from machina.utils import get_device
from machina import loss_functional as lf
from machina.traj import Traj


def compute_vs(data, vf):
    """
    Computing Value Function.

    Parameters
    ----------
    data : Traj or epis(dict of ndarray)
    vf : SVFunction

    Returns
    -------
    data : Traj or epi(dict of ndarray)
        Corresponding to input
    """
    if isinstance(data, Traj):
        epis = data.current_epis
    else:
        epis = data

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
    """
    Set prioritization to all episodes.

    Parameters
    ----------
    data : Traj or epis(dict of ndarray)
    pri : torch.Tensor

    Returns
    -------
    data : Traj or epi(dict of ndarray)
        Corresponding to input
    """
    if isinstance(data, Traj):
        epis = data.current_epis
    else:
        epis = data

    for epi in epis:
        pris = pri.repeat(len(epi['obs']))
        epi['pris'] = pris.cpu().numpy()

    return data


def compute_pris(data, qf, targ_qf, pol, gamma, continuous=True, deterministic=True, rnn=False, sampling=1, alpha=0.6, epsilon=1e-6):
    """
    Compute prioritization.

    Parameters
    ----------
    data : Traj or epis(dict of ndarray)
    qf : SAVfunction
    targ_qf : SAVfunction
    pol : Pol
    gamma : float
    continuous : bool
    deterministic : bool
    rnn : bool
    sampling : int
    alpha : float
    epsilen : float

    Returns
    -------
    data : Traj or epi(dict of ndarray)
        Corresponding to input
    """
    if continuous:
        if isinstance(data, Traj):
            epis = data.current_epis
        else:
            epis = data
        for epi in epis:
            data_map = dict()
            keys = ['obs', 'acs', 'rews', 'next_obs', 'dones']
            for key in keys:
                data_map[key] = torch.tensor(epi[key], device=get_device())
            if rnn:
                qf.reset()
                targ_qf.reset()
                pol.reset()
                keys = ['obs', 'acs', 'next_obs']
                for key in keys:
                    data_map[key] = data_map[key].unsqueeze(1)
            with torch.no_grad():
                bellman_loss = lf.bellman(
                    qf, targ_qf, pol, data_map, gamma, continuous, deterministic, sampling, reduction='none')
                td_loss = torch.sqrt(bellman_loss*2)
                pris = (torch.abs(td_loss) + epsilon) ** alpha
                epi['pris'] = pris.cpu().numpy()
        return data
    else:
        raise NotImplementedError(
            "Only Q function with continuous action space is supported now.")


def compute_seq_pris(data, seq_length, eta=0.9):
    """
    Computing priorities of each sequence in episodes.

    Parameters
    ----------
    data : Traj or epis(dict of ndarray)
    seq_length : int
        Length of batch
    eta : float

    Returns
    -------
    data : Traj or epi(dict of ndarray)
        Corresponding to input
    """
    if isinstance(data, Traj):
        epis = data.current_epis
    else:
        epis = data

    for epi in epis:
        n_seq = len(epi['pris']) - seq_length + 1
        abs_pris = np.abs(epi['pris'])
        seq_pris = np.array([eta * np.max(abs_pris[i:i+seq_length]) + (1 - eta) *
                             np.mean(abs_pris[i:i+seq_length]) for i in range(n_seq)], dtype='float32')
        pad = np.zeros((seq_length - 1,), dtype='float32')
        epi['seq_pris'] = np.concatenate([seq_pris, pad])

    return data


def compute_rets(data, gamma):
    """
    Computing discounted cumulative returns.

    Parameters
    ----------
    data : Traj or epis(dict of ndarray)
    gamma : float
        Discount rate

    Returns
    -------
    data : Traj or epi(dict of ndarray)
        Corresponding to input
    """
    if isinstance(data, Traj):
        epis = data.current_epis
    else:
        epis = data

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
    data : Traj or epis(dict of ndarray)
    gamma : float
        Discount rate
    lam : float
        Bias-Variance trade-off parameter

    Returns
    -------
    data : Traj or epi(dict of ndarray)
        Corresponding to input
    """
    if isinstance(data, Traj):
        epis = data.current_epis
    else:
        epis = data

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


def compute_hs(data, func, hs_name='hs', input_acs=False):
    """
    Computing Hidden State of RNN Cell.

    Parameters
    ----------
    data : Traj or epis(dict of ndarray)
    func : 
        Any function. for example pols, vf and qf.

    Returns
    -------
    data : Traj or epi(dict of ndarray)
        Corresponding to input
    """
    if isinstance(data, Traj):
        epis = data.current_epis
    else:
        epis = data

    func.reset()
    with torch.no_grad():
        for epi in epis:
            obs = torch.tensor(
                epi['obs'], dtype=torch.float, device=get_device()).unsqueeze(1)
            time_seq = obs.size()[0]
            if input_acs:
                acs = torch.tensor(
                    epi['acs'], dtype=torch.float, device=get_device()).unsqueeze(1)
                hs_seq = [func(obs[i:i+1], acs[i:i+1])[-1]['hs']
                          for i in range(time_seq)]
            else:
                hs_seq = [func(obs[i:i+1])[-1]['hs'] for i in range(time_seq)]
            if isinstance(hs_seq[0], tuple):
                hs = np.array([[h.squeeze().detach().cpu().numpy()
                                for h in hs] for hs in hs_seq], dtype='float32')
            else:
                hs = np.array(hs.detach().cpu().numpy(), dtype='float32')
            epi[hs_name] = hs

    return data


def centerize_advs(data, eps=1e-6):
    """
    Centerizing Advantage Function.

    Parameters
    ----------
    data : Traj or epis(dict of ndarray)
    eps : float
        Small value for preventing 0 division.

    Returns
    -------
    data : Traj or epi(dict of ndarray)
        Corresponding to input
    """
    if isinstance(data, Traj):
        epis = data.current_epis
    else:
        epis = data

    _advs = np.concatenate([epi['advs'] for epi in epis])
    for epi in epis:
        epi['advs'] = (epi['advs'] - np.mean(_advs)) / (np.std(_advs) + eps)

    return data


def add_next_obs(data):
    """
    Adding next observations to episodes.

    Parameters
    ----------
    data : Traj or epis(dict of ndarray)

    Returns
    -------
    data : Traj or epi(dict of ndarray)
        Corresponding to input
    """
    if isinstance(data, Traj):
        epis = data.current_epis
    else:
        epis = data

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
    data : Traj or epis(dict of ndarray)

    Returns
    -------
    data : Traj or epi(dict of ndarray)
        Corresponding to input
    """
    if isinstance(data, Traj):
        epis = data.current_epis
    else:
        epis = data

    for epi in epis:
        h_masks = np.zeros_like(epi['rews'])
        h_masks[0] = 1
        epi['h_masks'] = h_masks

    return data


def compute_pseudo_rews(data, rew_giver, state_only=False):
    if isinstance(data, Traj):
        epis = data.current_epis
    else:
        epis = data

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


def compute_diayn_rews(data, rew_giver):
    epis = data.current_epis
    for epi in epis:
        obs = torch.as_tensor(
            epi['obs'], dtype=torch.float, device=get_device())
        with torch.no_grad():
            rews, info = rew_giver(obs)
        epi['rews'] = rews.cpu().numpy()
    return data


def train_test_split(epis, train_size):
    num_epi = len(epis)
    num_train = int(num_epi * train_size)
    indices = np.arange(num_epi)
    train_epis, test_epis = [[epis[indice] for indice in indices] for indices in
                             np.array_split(indices, [num_train])]

    return train_epis, test_epis


def normalize_obs_and_acs(data, mean_obs=None, std_obs=None, mean_acs=None, std_acs=None, return_statistic=True, eps=1e-6):
    with torch.no_grad():
        if isinstance(data, Traj):
            epis = data.current_epis
        else:
            epis = data
        obs = []
        acs = []
        for epi in epis:
            obs.extend(epi['obs'])
            acs.extend(epi['acs'])
        obs = np.array(obs, dtype=np.float32)
        acs = np.array(acs, dtype=np.float32)

        if mean_obs is None:
            mean_obs = np.mean(obs, axis=0, keepdims=True)
        if std_obs is None:
            std_obs = np.std(
                obs, axis=0, keepdims=True) + eps
        if mean_acs is None:
            mean_acs = np.mean(acs, axis=0, keepdims=True)
        if std_acs is None:
            std_acs = np.std(
                acs, axis=0, keepdims=True) + eps

        for epi in epis:
            epi['obs'] = (epi['obs'] - mean_obs) / std_obs
            epi['acs'] = (epi['acs'] - mean_acs) / std_acs
            epi['next_obs'] = (
                epi['next_obs'] - mean_obs) / std_obs

    if return_statistic:
        return data, mean_obs, std_obs, mean_acs, std_acs
    else:
        return data

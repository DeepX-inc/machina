"""
These are functions for loss.
Algorithms should be written by combining these functions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from machina.utils import detach_tensor_dict, get_device


def pg_clip(pol, batch, clip_param, ent_beta):
    """
    Policy Gradient with clipping.

    Parameters
    ----------
    pol : Pol
    batch : dict of torch.Tensor
    clip_param : float
    ent_beta : float
        entropy coefficient

    Returns
    -------
    pol_loss : torch.Tensor
    """
    obs = batch['obs']
    acs = batch['acs']
    advs = batch['advs']

    if pol.rnn:
        h_masks = batch['h_masks']
        out_masks = batch['out_masks']
    else:
        h_masks = None
        out_masks = torch.ones_like(advs)

    pd = pol.pd

    old_llh = pd.llh(
        batch['acs'],
        batch,
    )

    pol.reset()
    _, _, pd_params = pol(obs, h_masks=h_masks)

    new_llh = pd.llh(acs, pd_params)
    ratio = torch.exp(new_llh - old_llh)
    pol_loss1 = - ratio * advs
    pol_loss2 = - torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advs
    pol_loss = torch.max(pol_loss1, pol_loss2)
    pol_loss = torch.mean(pol_loss * out_masks)

    ent = pd.ent(pd_params)
    pol_loss -= ent_beta * torch.mean(ent)

    return pol_loss


def pg_kl(pol, batch, kl_beta, ent_beta=0):
    """
    Policy Gradient with KL divergence restriction.

    Parameters
    ----------
    pol : Pol
    batch : dict of torch.Tensor
    kl_beta : float
        KL divergence coefficient

    Returns
    -------
    pol_loss : torch.Tensor
    """
    obs = batch['obs']
    acs = batch['acs']
    advs = batch['advs']

    if pol.rnn:
        h_masks = batch['h_masks']
        out_masks = batch['out_masks']
    else:
        h_masks = None
        out_masks = torch.ones_like(advs)

    pd = pol.pd

    old_llh = pol.pd.llh(
        batch['acs'],
        batch
    )

    pol.reset()
    _, _, pd_params = pol(obs, h_masks=h_masks)

    new_llh = pol.pd.llh(acs, pd_params)
    ratio = torch.exp(new_llh - old_llh)
    pol_loss = ratio * advs * out_masks

    kl = pol.pd.kl_pq(
        batch,
        pd_params
    )

    pol_loss -= kl_beta * kl * out_masks
    pol_loss = - torch.mean(pol_loss)

    ent = pd.ent(pd_params)
    pol_loss -= ent_beta * torch.mean(ent)
    return pol_loss


def bellman(qf, targ_qf, targ_pol, batch, gamma, continuous=True, deterministic=True, sampling=1, reduction='elementwise_mean'):
    """
    Bellman loss.
    Mean Squared Error of left hand side and right hand side of Bellman Equation.

    Parameters
    ----------
    qf : SAVfunction
    targ_qf : SAVfunction
    targ_pol : Pol
    batch : dict of torch.Tensor
    gamma : float
    continuous : bool
        action space is continuous or not
    sampling : int
        Number of samping in calculating expectation.
    reduction : str
      This argument takes only elementwise, sum, and none.
      Loss shape is pytorch's manner.

    Returns
    -------
    bellman_loss : torch.Tensor
    """
    if continuous:
        obs = batch['obs']
        acs = batch['acs']
        rews = batch['rews']
        next_obs = batch['next_obs']
        dones = batch['dones']

        targ_pol.reset()
        _, _, pd_params = targ_pol(next_obs)
        pd = targ_pol.pd

        next_acs = pd.sample(pd_params, torch.Size([sampling]))
        next_obs = next_obs.expand([sampling] + list(next_obs.size()))
        targ_q, _ = targ_qf(next_obs, next_acs)
        next_q = torch.mean(targ_q, dim=0)

        targ = rews + gamma * next_q * (1 - dones)
        targ = targ.detach()
        q, _ = qf(obs, acs)

        ret = 0.5 * (q - targ)**2
        if reduction != 'none':
            ret = torch.mean(
                ret) if reduction == 'elementwise_mean' else torch.sum(ret)
        return ret
    else:
        raise NotImplementedError(
            "Only Q function with continuous action space is supported now.")


def sac(pol, qf, targ_qf, log_alpha, batch, gamma, sampling, normalize=False, eps=1e-6):
    """
    Loss for soft actor critic.

    Parameters
    ----------
    pol : Pol
    qf : SAVfunction
    targ_qf : SAVfunction
    log_alpha : torch.Tensor
    batch : dict of torch.Tensor
    gamma : float
    sampling : int
        Number of samping in calculating expectation.
    normalize : bool
        If True, normalize value of log likelihood.
    eps : float

    Returns
    -------
    pol_loss, qf_loss, alpha_loss : torch.Tensor, torch.Tensor, torch.Tensor
    """
    obs = batch['obs']
    acs = batch['acs']
    rews = batch['rews']
    next_obs = batch['next_obs']
    dones = batch['dones']

    alpha = torch.exp(log_alpha)

    pol.reset()
    _, _, pd_params = pol(obs)
    pol.reset()
    _, _, next_pd_params = pol(next_obs)
    pd = pol.pd

    sampled_obs = obs.expand([sampling] + list(obs.size()))
    sampled_next_obs = next_obs.expand([sampling] + list(next_obs.size()))

    sampled_acs = pd.sample(pd_params, torch.Size([sampling]))
    sampled_next_acs = pd.sample(next_pd_params, torch.Size([sampling]))

    sampled_llh = pd.llh(sampled_acs.detach(), pd_params)
    sampled_next_llh = pd.llh(sampled_next_acs, next_pd_params)

    sampled_q, _ = qf(sampled_obs, sampled_acs)
    sampled_next_targ_q, _ = targ_qf(sampled_next_obs, sampled_next_acs)

    next_v = torch.mean(sampled_next_targ_q - alpha * sampled_next_llh, dim=0)

    q_targ = rews + gamma * next_v * (1 - dones)
    q_targ = q_targ.detach()

    q, _ = qf(obs, acs)

    qf_loss = 0.5 * torch.mean((q - q_targ)**2)

    pg_weight = (alpha * sampled_llh - sampled_q).detach()

    if normalize:
        pg_weight = (pg_weight - pg_weight.mean()) / (pg_weight.std() + eps)

    pol_loss = torch.mean(sampled_llh * pg_weight)

    alpha_loss = - torch.mean(log_alpha * (sampled_llh -
                                           np.prod(pol.ac_space.shape).item()).detach())

    return pol_loss, qf_loss, alpha_loss


def ag(pol, qf, batch, sampling=1):
    """
    DDPG style action gradient.

    Parameters
    ----------
    pol : Pol
    qf : SAVfunction
    batch : dict of torch.Tensor
    sampling : int
        Number of samping in calculating expectation.

    Returns
    -------
    pol_loss : torch.Tensor
    """
    obs = batch['obs']

    _, _, pd_params = pol(obs)
    pd = pol.pd

    acs = pd.sample(pd_params, torch.Size([sampling]))
    q, _ = qf(obs.expand([sampling] + list(obs.size())), acs)
    q = torch.mean(q, dim=0)

    pol_loss = - torch.mean(q)

    return pol_loss


def pg(pol, batch, ent_beta=0):
    """
    Policy Gradient.

    Parameters
    ----------
    pol : Pol
    batch : dict of torch.Tensor

    Returns
    -------
    pol_loss : torch.Tensor
    """
    obs = batch['obs']
    acs = batch['acs']
    advs = batch['advs']

    pd = pol.pd
    pol.reset()
    if pol.rnn:
        h_masks = batch['h_masks']
        out_masks = batch['out_masks']
        _, _, pd_params = pol(obs, h_masks=h_masks)
    else:
        out_masks = torch.ones_like(advs)
        _, _, pd_params = pol(obs)

    llh = pol.pd.llh(acs, pd_params)

    pol_loss = - torch.mean(llh * advs * out_masks)
    ent = pd.ent(pd_params)
    pol_loss -= ent_beta * torch.mean(ent)
    return pol_loss


def monte_carlo(vf, batch, clip_param=0.2, clip=False):
    """
    Montecarlo loss for V function.

    Parameters
    ----------
    vf : SVfunction
    batch : dict of torch.Tensor
    clip_param : float
    clip : bool

    Returns
    -------

    """
    obs = batch['obs']
    rets = batch['rets']

    vf.reset()
    if vf.rnn:
        h_masks = batch['h_masks']
        out_masks = batch['out_masks']
        vs, _ = vf(obs, h_masks=h_masks)
    else:
        out_masks = torch.ones_like(rets)
        vs, _ = vf(obs)

    vfloss1 = (vs - rets)**2
    if clip:
        old_vs = batch['vs']
        vpredclipped = old_vs + \
            torch.clamp(vs - old_vs, -clip_param, clip_param)
        vfloss2 = (vpredclipped - rets)**2
        vf_loss = 0.5 * torch.mean(torch.max(vfloss1, vfloss2) * out_masks)
    else:
        vf_loss = 0.5 * torch.mean(vfloss1 * out_masks)
    return vf_loss


def dynamics(dm, batch, target='next_obs', td=True):
    """
    MSE loss for Dynamics models.
    Parameters
    ----------
    dm : Dynamics Model
    batch : dict of torch.Tensor
    target : string
        Prediction target is next_obs or rews.
    td : bool
        If True, the dynamics model learn temporal difference of dynamics.

    Returns
    -------
    model_loss : torch.Tensor
    """

    obs = batch['obs']
    acs = batch['acs']

    dm.reset()
    if dm.rnn:
        h_masks = batch['h_masks']
        out_masks = batch['out_masks']
        pred, _ = dm(obs, acs, h_masks=h_masks)
    else:
        out_masks = torch.ones_like(obs)
        pred, _ = dm(obs, acs)

    if target == 'rews' or not td:
        dm_loss = (pred - batch[target])**2
    else:
        dm_loss = (pred - (batch['next_obs'] - batch['obs']))**2
    dm_loss = 0.5 * torch.mean(dm_loss * out_masks)

    return dm_loss


def likelihood(pol, batch):
    obs = batch['obs']
    acs = batch['acs']
    _, _, pd_params = pol(obs)
    llh = pol.pd.llh(acs, pd_params)
    pol_loss = -torch.mean(llh)
    return pol_loss


def cross_ent(discrim, batch, expert_or_agent, ent_beta):
    obs = batch['obs']
    acs = batch['acs']
    len = obs.shape[0]
    logits, _ = discrim(obs, acs)
    discrim_loss = F.binary_cross_entropy_with_logits(
        logits, torch.ones(len, device=get_device())*expert_or_agent)
    ent = (1 - torch.sigmoid(logits))*logits - F.logsigmoid(logits)
    discrim_loss -= ent_beta * torch.mean(ent)
    return discrim_loss

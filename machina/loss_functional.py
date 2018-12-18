# Copyright 2018 DeepX Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import torch
import torch.nn as nn


def pg_clip(pol, batch, clip_param, ent_beta):
    obs = batch['obs']
    acs = batch['acs']
    advs = batch['advs']

    if pol.rnn:
        h_masks = batch['h_masks']
        out_masks = batch['out_masks']
    else:
        out_masks = torch.ones_like(advs)

    pd = pol.pd

    old_llh = pd.llh(
        batch['acs'],
        batch,
    )

    pol.reset()
    if pol.rnn:
        _, _, pd_params = pol(obs, h_masks=h_masks)
    else:
        _, _, pd_params = pol(obs)

    new_llh = pd.llh(acs, pd_params)
    ratio = torch.exp(new_llh - old_llh)
    pol_loss1 = - ratio * advs
    pol_loss2 = - torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advs
    pol_loss = torch.max(pol_loss1, pol_loss2)
    pol_loss = torch.mean(pol_loss * out_masks)

    ent = pd.ent(pd_params)
    pol_loss -= ent_beta * torch.mean(ent)

    return pol_loss

def pg_kl(pol, batch, kl_beta):
    obs = batch['obs']
    acs = batch['acs']
    advs = batch['advs']

    if pol.rnn:
        h_masks = batch['h_masks']
        out_masks = batch['out_masks']
    else:
        out_masks = torch.ones_like(advs)

    pd = pol.pd

    old_llh = pol.pd.llh(
        batch['acs'],
        batch
    )

    pol.reset()
    if pol.rnn:
        _, _, pd_params = pol(obs, h_masks=h_masks)
    else:
        _, _, pd_params = pol(obs)

    new_llh = pol.pd.llh(acs, pd_params)
    ratio = torch.exp(new_llh - old_llh)
    pol_loss = ratio * advs * out_masks

    kl = pol.pd.kl_pq(
        batch,
        pd_params
    )

    pol_loss -= kl_beta * kl * out_masks
    pol_loss = - torch.mean(pol_loss)

    return pol_loss

def dpg(pol, qf, batch):
    obs = batch['obs']

    _, _, param = pol(obs)

    q, _ = qf(obs, param['mean'])
    pol_loss = -torch.mean(q)

    return pol_loss

def bellman(qf, targ_qf, targ_pol, batch, gamma, continuous=True, deterministic=True, sampling=1):
    if continuous:
        obs = batch['obs']
        acs = batch['acs']
        rews = batch['rews']
        next_obs = batch['next_obs']
        dones = batch['dones']

        # TODO
        if deterministic:
            _, _, param = targ_pol(next_obs)
            next_q, _ = targ_qf(next_obs, param['mean'])
        else:
            next_q = 0
            _, _, pd_params = pol(next_obs)
            next_means, next_log_stds = pd_params['mean'], pd_params['log_std']
            for _ in range(sampling):
                next_acs = next_means + torch.randn_like(next_means) * torch.exp(next_log_stds)
                _next_q, _ = targ_qf(next_obs, next_acs)
                next_q += _next_q
            next_q /= sampling

        targ = rews + gamma * next_q * (1 - dones)
        targ = targ.detach()
        q, _ = qf(obs, acs)

        return 0.5 * torch.mean((q - targ)**2)
    else:
        raise NotImplementedError()

def sac(pol, qf, vf, batch, sampling):
    obs = batch['obs']

    pol_loss = 0
    _, _, pd_params = pol(obs)
    # TODO: fast sampling
    for _ in range(sampling):
        acs = pol.pd.sample(pd_params)
        llh = pol.pd.llh(acs.detach(), pd_params)

        q, _ = qf(obs, acs)
        q = q.detach()

        v, _ = vf(obs)
        v = v.detach()

        pol_loss += llh * (llh.detach() - q + v)
    pol_loss /= sampling

    return torch.mean(pol_loss)

def sac_sav(qf, vf, batch, gamma):
    obs = batch['obs']
    acs = batch['acs']
    rews = batch['rews']
    next_obs = batch['next_obs']
    dones = batch['dones']

    v, _ = vf(next_obs)

    targ = rews + gamma * v * (1 - dones)
    targ = targ.detach()

    q, _ = qf(obs, acs)

    return 0.5 * torch.mean((q - targ)**2)

def sac_sv(pol, qf, vf, batch, sampling):
    obs = batch['obs']

    targ = 0
    _, _, pd_params = pol(obs)
    # TODO: fast sampling
    for _ in range(sampling):
        acs = pol.pd.sample(pd_params)
        llh = pol.pd.llh(acs, pd_params)
        q, _ = qf(obs, acs)
        targ += q - llh
    targ /= sampling
    targ = targ.detach()

    v, _ = vf(obs)

    return 0.5 * torch.mean((v - targ)**2)

def svg(pol, qf, batch, sampling, kl_coeff=0):
    obs = batch['obs']

    q = 0
    _, _, pd_params = pol(obs)
    # TODO: fast sampling
    for _ in range(sampling):
        acs = pol.pd.sample(pd_params)
        _q, _ = qf(obs, acs)
        q += _q
    q /= sampling

    pol_loss = - torch.mean(q)

    _, _, pd_params = pol(obs)
    kl = pol.pd.kl_pq(
        {k:d.detach() for k, d in pd_params.items()},
        pd_params
    )
    mean_kl = torch.mean(kl)

    return pol_loss + mean_kl * kl_coeff

def pg(pol, batch, volatile=False):
    obs = batch['obs']
    acs = batch['acs']
    advs = batch['advs']

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
    return pol_loss

def monte_carlo(vf, batch, clip_param=0.2, clip=False):
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
        vpredclipped = old_vs + torch.clamp(vs - old_vs, -clip_param, clip_param)
        vfloss2 = (vpredclipped - rets)**2
        vf_loss = 0.5 * torch.mean(torch.max(vfloss1, vfloss2) * out_masks)
    else:
        vf_loss = 0.5 * torch.mean(vfloss1 * out_masks)
    return vf_loss

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
#
# This is an implementation of Trust Region Policy Optimization.
# See https://arxiv.org/abs/1502.05477
#


import torch
import torch.nn as nn
import numpy as np

from machina.misc import logger
from machina.utils import detach_tensor_dict

def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def linesearch(
        pol,
        batch,
        f,
        x,
        fullstep,
        expected_improve_rate,
        max_backtracks=10,
        accept_ratio=.1
    ):
    fval = f(pol, batch, True).detach()
    print("loss before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        nn.utils.vector_to_parameters(xnew, pol.parameters())
        newfval = f(pol, batch, True).detach()
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
        #if actual_improve[0] > 0:
            print("loss after", newfval.item())
            return True, xnew
    return False, x

def make_pol_loss(pol, batch, volatile=False):
    obs = batch['obs']
    acs = batch['acs']
    advs = batch['advs']

    if pol.rnn:
        init_hs = batch['init_hs']
        masks = batch['dones']
        _, _, pd_params = pol(obs, init_hs, masks)
    else:
        _, _, pd_params = pol(obs)

    llh = pol.pd.llh(acs, pd_params)

    pol_loss = - torch.mean(llh * advs)
    return pol_loss

def make_kl(pol, batch):
    obs = batch['obs']
    if pol.rnn:
        init_hs = batch['init_hs']
        masks = batch['dones']
        _, _, pd_params = pol(obs, init_hs, masks)
    else:
        _, _, pd_params = pol(obs)

    return pol.pd.kl_pq(
        detach_tensor_dict(pd_params),
        pd_params
    )

def update_pol(pol, batch, make_pol_loss=make_pol_loss, make_kl=make_kl, max_kl=0.01, damping=0.1, num_cg=10):
    pol_loss = make_pol_loss(pol, batch)
    grads = torch.autograd.grad(pol_loss, pol.parameters(), create_graph=True)
    grads = [g.contiguous() for g in grads]
    flat_pol_loss_grad = nn.utils.parameters_to_vector(grads).detach()

    def Fvp(v):
        kl = make_kl(pol, batch)
        kl = torch.mean(kl)

        grads = torch.autograd.grad(kl, pol.parameters(), create_graph=True)
        grads = [g.contiguous() for g in grads]
        flat_grad_kl = nn.utils.parameters_to_vector(grads)
        gvp = torch.sum(flat_grad_kl * v)
        grads = torch.autograd.grad(gvp, pol.parameters())
        grads = [g.contiguous() for g in grads]
        fvp = nn.utils.parameters_to_vector(grads).detach()

        return fvp + v * damping

    stepdir = conjugate_gradients(Fvp, -flat_pol_loss_grad, num_cg)

    shs = 0.5 * torch.sum(stepdir * Fvp(stepdir), 0, keepdim=True)
    if (shs < 0).any():
        logger.log('invalid shs')
        return pol_loss.data.cpu().numpy()

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdotstepdir = torch.sum(-flat_pol_loss_grad * stepdir, 0, keepdim=True)

    prev_params = nn.utils.parameters_to_vector([p.contiguous() for p in pol.parameters()]).detach()
    success, new_params = linesearch(pol, batch, make_pol_loss, prev_params, fullstep,
                                     neggdotstepdir / lm[0])
    nn.utils.vector_to_parameters(new_params, pol.parameters())

    return pol_loss.detach().cpu().numpy()

def make_vf_loss(vf, batch):
    obs = batch['obs']
    rets = batch['rets']

    if vf.rnn:
        init_hs = batch['init_hs']
        masks = batch['dones']
        vs, _ = vf(obs, init_hs, masks)
    else:
        vs, _ = vf(obs)

    vf_loss = 0.5 * torch.mean((vs - rets)**2)
    return vf_loss

def update_vf(vf, optim_vf, batch):
    vf_loss = make_vf_loss(vf, batch)
    optim_vf.zero_grad()
    vf_loss.backward()
    optim_vf.step()
    return vf_loss.detach().cpu().numpy()

def train(data, pol, vf,
        optim_vf,
        epoch=5, batch_size=64,# optimization hypers
        max_kl=0.01, num_cg=10, damping=0.1,
        ):

    pol_losses = []
    vf_losses = []
    logger.log("Optimizing...")
    for batch in data.full_batch(1):
        pol_loss = update_pol(pol, batch, max_kl=max_kl, num_cg=num_cg, damping=damping)
        pol_losses.append(pol_loss)
        if 'Normalized' in vf.__class__.__name__:
            vf.set_mean(torch.mean(batch['rets'], 0, keepdim=True))
            vf.set_std(torch.std(batch['rets'], 0, keepdim=True))

    for batch in data.iterate(batch_size=batch_size, epoch=epoch):
        vf_loss = update_vf(vf, optim_vf, batch)
        vf_losses.append(vf_loss)

    logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses, VfLoss=vf_losses)



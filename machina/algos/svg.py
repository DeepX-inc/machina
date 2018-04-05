import torch
import torch.nn as nn
from machina.utils import Variable, torch2torch
from machina.misc import logger

def make_pol_loss(pol, qf, batch, sampling, kl_coeff=0):
    obs = Variable(batch['obs'])

    q = 0
    _, _, pd_params = pol(obs)
    for _ in range(sampling):
        acs = pol.pd.sample(pd_params)
        q += qf(obs, acs)
    q /= sampling

    pol_loss = -torch.mean(q)

    _, _, pd_params = pol(obs)
    kl = pol.pd.kl_pq(
        {k:Variable(d.data) for k, d in pd_params.items()},
        pd_params
    )
    mean_kl = torch.mean(kl)

    return pol_loss + mean_kl * kl_coeff

def update_pol(pol, qf, optim_pol, batch, sampling):
    pol_loss = make_pol_loss(pol, qf, batch, sampling)
    optim_pol.zero_grad()
    pol_loss.backward()
    optim_pol.step()
    return pol_loss.data.cpu().numpy()

def make_bellman_loss(qf, targ_qf, pol, batch, gamma, sampling):
    obs = Variable(batch['obs'])
    acs = Variable(batch['acs'])
    rews = Variable(batch['rews'])
    next_obs = Variable(batch['next_obs'])
    terminals = Variable(batch['terminals'])
    expected_next_q = 0
    _, _, pd_params = pol(next_obs)
    next_means, next_log_stds = pd_params['mean'], pd_params['log_std']
    for _ in range(sampling):
        next_acs = next_means + Variable(torch2torch(torch.randn(next_means.size()))) * torch.exp(next_log_stds)
        expected_next_q += targ_qf(next_obs, next_acs)
    expected_next_q /= sampling
    targ = rews + gamma * expected_next_q * (1 - terminals)
    targ = Variable(targ.data)

    return 0.5 * torch.mean((qf(obs, acs) - targ)**2)

def make_mc_loss(qf, batch):
    obs = Variable(batch['obs'])
    acs = Variable(batch['acs'])
    rets = Variable(batch['rets'])
    return 0.5 * torch.mean((qf(obs, acs) - rets)**2)

def train(off_data,
        pol, qf, targ_qf,
        optim_pol, optim_qf,
        epoch, batch_size,# optimization hypers
        tau, gamma, # advantage estimation
        sampling,
        ):

    pol_losses = []
    qf_losses = []
    logger.log("Optimizing...")
    for batch in off_data.iterate(batch_size, epoch):
        qf_bellman_loss = make_bellman_loss(qf, targ_qf, pol, batch, gamma, sampling)
        optim_qf.zero_grad()
        qf_bellman_loss.backward()
        optim_qf.step()

        pol_loss = make_pol_loss(pol, qf, batch, sampling)
        optim_pol.zero_grad()
        pol_loss.backward()
        optim_pol.step()

        for p, targ_p in zip(qf.parameters(), targ_qf.parameters()):
            targ_p.data.copy_((1 - tau) * targ_p.data + tau * p.data)
        qf_losses.append(qf_bellman_loss.data.cpu().numpy())
        pol_losses.append(pol_loss.data.cpu().numpy())

    logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses,
        QfLoss=qf_losses,
    )


"""
This is an implementation of Deep Planning Network.
See https://arxiv.org/abs/1811.04551
"""

import torch
import torch.nn as nn
import numpy as np

from machina import loss_functional as lf
from machina.utils import detach_tensor_dict, get_device
from machina import logger


def train(traj, rssm, ob_model, rew_model, optim_rssm, optim_om, optim_rm, scheduler_rssm, scheduler_om, scheduler_rm, epoch=60, pred_steps=50, max_latend_pred_steps=50, batch_size=50, num_epi_per_seq=1):
    """
    Train function for dynamics model.

    Parameters
    ----------
    traj : Traj
        On policy trajectory.
    dyn_model : Model
        dynamics model.
    optim_dm : torch.optim.Optimizer
        Optimizer for dynamics model.
    epoch : int
        Number of iteration.
    batch_size : int
        Number of batches.
    target : str
        Target of prediction is next_obs or rews.
    td : bool
        If True, dyn_model learn temporal differance of target.
    num_epi_per_seq : int
        Number of episodes in one sequence for rnn.

    Returns
    -------
    result_dict : dict
        Dictionary which contains losses information.
    """

    losses = []
    obs_losses = []
    rews_losses = []
    recun_losses = []
    divergence_losses = []

    reward_loss_scale = 10.0
    overshooting_reward_loss_scale = 100.0
    global_divergence_loss_scale = 0.1

    logger.log("Optimizing...")

    iterator = traj.random_batch_rnn(
        batch_size, seq_length=pred_steps, epoch=epoch)
    for batch in iterator:
        # embed obs
        embedded = []
        for obs in batch['obs']:
            embedded.append(rssm.encode(obs).unsqueeze(0))
        batch['embedded_obs'] = torch.cat(embedded, dim=0)

        # overshooting
        """
        Example: 
        posterior
            [t, t+1, t+2, t+3, t+4, t+5]
        prior
            [
                [ - , t+1, t+2, t+3, -  ,  - ],
                [ - ,  - , t+2, t+3, t+4,  - ],
                [ - ,  - ,  - , t+3, t+4, t+5],
                [ - ,  - ,  - ,  - , t+4, t+5],
                [ - ,  - ,  - ,  - ,  - , t+5],
            ]
        """
        rssm.reset()
        posteriors = [None for _ in range(pred_steps)]
        priors = [[None for _ in range(pred_steps)]
                  for _ in range(pred_steps-1)]

        zero_state = torch.zeros(
            batch_size, rssm.state_size, dtype=torch.float, device=get_device())
        zero_action = torch.zeros(batch_size, batch['acs'].size(
        )[-1], dtype=torch.float, device=get_device())
        posteriors[0] = rssm.posterior(
            zero_state, zero_action, batch['embedded_obs'][0])

        for t in range(pred_steps-1):
            posteriors[t+1] = rssm.posterior(posteriors[t]['sample'], batch['acs']
                                             [t], batch['embedded_obs'][t+1], hs=posteriors[t]['belief'])
            priors[t][t+1] = rssm.prior(posteriors[t]['sample'],
                                        batch['acs'][t], hs=posteriors[t]['belief'])
            for prior_index in range(t):
                priors[prior_index][t+1] = rssm.prior(
                    priors[prior_index][t]['sample'], batch['acs'][t], hs=priors[prior_index][t]['belief'])

        # compute loss
        loss = 0
        sum_obs_loss = 0
        sum_rews_loss = 0
        sum_recun_loss = 0
        sum_divergence_loss = 0
        for t in range(pred_steps-1):
            # zero step recunstruction loss
            features = rssm.features_from_state(posteriors[t])
            pred_obs, obs_dict = ob_model(features, acs=None)
            pred_rews, rews_dict = rew_model(
                features, acs=None)
            obs_loss = 0.5 * \
                ((obs_dict['mean'] - batch['obs'][t]) ** 2)
            rews_loss = 0.5 * ((rews_dict['mean'] - batch['rews'][t]) ** 2)
            obs_loss = torch.mean(obs_loss)
            rews_loss = torch.mean(rews_loss) * reward_loss_scale
            sum_obs_loss += obs_loss
            sum_rews_loss += rews_loss
            recun_loss = (obs_loss + rews_loss)

            # global divergence loss
            posterior_params = {
                'mean': posteriors[t]['mean'], 'log_std': posteriors[t]['log_std']}
            global_prior_params = {
                'mean': torch.zeros_like(posteriors[t]['mean'], device=get_device()),
                'log_std': torch.ones_like(posteriors[t]['log_std'], device=get_device())
            }
            global_kl = rssm.pd.kl_pq(
                posterior_params, global_prior_params)
            global_divergence_loss = torch.mean(
                global_kl, dim=0) * global_divergence_loss_scale

            # latent overshooting loss
            latent_rews_loss = 0
            divergence_loss = 0
            latend_pred_steps = min(max_latend_pred_steps, pred_steps-1-t)
            for d in range(1, latend_pred_steps+1):
                # overshooting reconstruction loss
                features = rssm.features_from_state(priors[t][t+d])
                pred_obs, obs_dict = ob_model(
                    features, acs=None)
                pred_rews, rews_dict = rew_model(
                    features, acs=None)
                rews_loss = 0.5 * \
                    ((rews_dict['mean'] - batch['rews'][t+d]) ** 2)
                rews_loss = torch.mean(rews_loss) * \
                    overshooting_reward_loss_scale
                if d == 1:
                    rews_loss *= max_latend_pred_steps
                latent_rews_loss += rews_loss

                # divergence loss
                posterior_params = {
                    'mean': posteriors[t+d]['mean'], 'log_std': posteriors[t+d]['log_std']}
                prior_params = {
                    'mean': priors[t][t+d]['mean'], 'log_std': priors[t][t+d]['log_std']}
                # free nats
                kl = rssm.pd.kl_pq(
                    posterior_params, prior_params).clamp(max=2.0)
                divergence_loss += torch.mean(kl,
                                              dim=0)
                if d == 1:
                    divergence_loss *= latend_pred_steps

            latent_rews_loss /= max_latend_pred_steps
            sum_rews_loss += latent_rews_loss
            recun_loss += latent_rews_loss
            divergence_loss /= max_latend_pred_steps
            divergence_loss += global_divergence_loss

            loss += recun_loss + divergence_loss
            sum_recun_loss += recun_loss
            sum_divergence_loss += divergence_loss

        # update
        optim_rssm.zero_grad()
        optim_om.zero_grad()
        optim_rm.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(ob_model.parameters(), 1000)
        nn.utils.clip_grad_norm_(rew_model.parameters(), 1000)
        nn.utils.clip_grad_norm_(rssm.parameters(), 1000)
        optim_rssm.step()
        optim_om.step()
        optim_rm.step()
        scheduler_rssm.step()
        scheduler_om.step()
        scheduler_rm.step()

        sum_obs_loss = torch.mean(sum_obs_loss).detach().cpu().numpy()
        sum_rews_loss = torch.mean(sum_rews_loss).detach().cpu().numpy()
        sum_recun_loss = torch.mean(sum_recun_loss).detach().cpu().numpy()
        sum_divergence_loss = torch.mean(
            sum_divergence_loss).detach().cpu().numpy()

        losses.append(loss.detach().cpu().numpy())
        obs_losses.append(sum_obs_loss)
        rews_losses.append(sum_rews_loss)
        recun_losses.append(sum_recun_loss)
        divergence_losses.append(sum_divergence_loss)

    logger.log("Optimization finished!")

    return dict(SumLoss=losses, ObsModelLoss=obs_losses, RewModelLoss=rews_losses,
                RecunstructionLoss=recun_losses, KLLoss=divergence_losses)

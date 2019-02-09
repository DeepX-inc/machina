import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina import logger


def update_pol(pol, optim_pol, batch, clip_param, ent_beta, max_grad_norm):
    """
    Update function for Policy.

    Parameters
    ----------
    pol : Pol
        Policy.
    optim_pol : torch.optim.Optimizer
        Optimizer for Policy.
    batch : dict
        Batch of trajectory
    clip_param : float
        Clipping ratio of objective function.
    ent_beta : float
        Entropy coefficient.
    max_grad_norm : float
        Maximum gradient norm.

    Returns
    -------
    pol_loss : ndarray
        Value of loss function.
    """
    pol_loss = lf.pg_clip(pol, batch, clip_param, ent_beta)
    optim_pol.zero_grad()
    pol_loss.backward()
    torch.nn.utils.clip_grad_norm_(pol.parameters(), max_grad_norm)
    optim_pol.step()
    return pol_loss.detach().cpu().numpy()


def update_vf(vf, optim_vf, batch, clip_param, clip, max_grad_norm):
    """
    Update function for V function.

    Parameters
    ----------
    vf : SVfunction
        V function.
    optim_vf : torch.optim.Optimizer
        Optimizer for V function.
    batch : dict
        Batch of trajectory
    clip_param : float
        Clipping ratio of objective function.
    clip: bool
        If True, vfunc is also updated by clipped objective function.
    max_grad_norm : float
        Maximum gradient norm.

    Returns
    -------
    vf_loss : ndarray
        Value of loss function.
    """
    vf_loss = lf.monte_carlo(vf, batch, clip_param, clip)
    optim_vf.zero_grad()
    vf_loss.backward()
    torch.nn.utils.clip_grad_norm_(vf.parameters(), max_grad_norm)
    optim_vf.step()
    return vf_loss.detach().cpu().numpy()


def update_model(model, optim_model, batch, target = 'next_obs'):
    """
    Update function for Dynamics model.

    Parameters
    ----------
    model : Model
        Dynamics model.
    optim_rf : torch.optim.Optimizer
        Optimizer for reward function.
    batch : dict
        Batch of trajectory
    target : string
        Target of prediction is next_obs or rews.

    Returns
    -------
    model_loss : ndarray
        Value of loss function.
    """
    model_loss = lf.dynamics(model, batch, target=target)
    optim_model.zero_grad()
    model_loss.backward()
    optim_model.step()
    return model_loss.detach().cpu().numpy()


def train_models(traj, ob_model, rew_model,
          optim_ob_model, optim_rew_model,
          epoch, batch_size
          ):
    """
    Train function for models.

    Parameters
    ----------
    traj : Traj
        On policy trajectory.
    ob_model : Model
        Observation model.
    rew_model : Model
        Reward model.
    optim_ob_model : torch.optim.Optimizer
        Optimizer for observation model.
    optim_rew_model : torch.optim.Optimizer
        Optimizer for reward model.
    epoch : int
        Number of iteration.
    batch_size : int
        Number of batches.

    Returns
    -------
    result_dict : dict
        Dictionary which contains losses information.
    """

    ob_model_losses = []
    rew_model_losses = []
    logger.log("Optimizing...")
    for batch in traj.random_batch(batch_size, epoch):
        ob_model_loss = update_model(ob_model, optim_ob_model, batch, target='next_obs')
        rew_model_loss = update_model(rew_model, optim_rew_model, batch, target='rews')

        ob_model_losses.append(ob_model_loss)
        rew_model_losses.append(rew_model_loss)
    logger.log("Optimization finished!")

    return dict(ObModelLoss=ob_model_losses, RewModelLoss=rew_model_losses)


def train_policy_and_vf(traj, pol, vf,
          optim_pol, optim_vf,
          epoch, batch_size, num_epi_per_seq=1,  # optimization hypers
          clip_param=0.2, ent_beta=1e-3,
          max_grad_norm=0.5,
          clip_vfunc=False
          ):
    """
    Train function for proximal policy optimization (clip).

    Parameters
    ----------
    traj : Traj
        On policy trajectory.
    pol : Pol
        Policy.
    vf : SVfunction
        V function.
    optim_pol : torch.optim.Optimizer
        Optimizer for Policy.
    optim_vf : torch.optim.Optimizer
        Optimizer for V function.
    epoch : int
        Number of iteration.
    batch_size : int
        Number of batches.
    num_epi_per_seq : int
        Number of episodes in one sequence for rnn.
    clip_param : float
        Clipping ratio of objective function.
    ent_beta : float
        Entropy coefficient.
    max_grad_norm : float
        Maximum gradient norm.
    clip_vfunc: bool
        If True, vfunc is also updated by clipped objective function.

    Returns
    -------
    result_dict : dict
        Dictionary which contains losses information.
    """

    pol_losses = []
    vf_losses = []
    logger.log("Optimizing...")
    iterator = traj.iterate(batch_size, epoch) if not pol.rnn else traj.iterate_rnn(
        batch_size=batch_size, num_epi_per_seq=num_epi_per_seq, epoch=epoch)
    for batch in iterator:
        pol_loss = update_pol(pol, optim_pol, batch,
                              clip_param, ent_beta, max_grad_norm)
        vf_loss = update_vf(vf, optim_vf, batch, clip_param,
                            clip_vfunc, max_grad_norm)

        pol_losses.append(pol_loss)
        vf_losses.append(vf_loss)
    logger.log("Optimization finished!")

    return dict(PolLoss=pol_losses, VfLoss=vf_losses)

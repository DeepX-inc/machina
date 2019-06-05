"""
This is an implementation of QT-Opt.
https://arxiv.org/abs/1806.10293
"""

from machina import loss_functional as lf
from machina import logger


def train(traj,
          qf, lagged_qf, targ_qf1, targ_qf2,
          optim_qf,
          epoch, batch_size,  # optimization hypers
          tau=0.9999, gamma=0.9,  # advantage estimation
          loss_type='mse',
          log_enable=True,
          ):
    """
    Train function for qtopt

    Parameters
    ----------
    traj : Traj
        Off policy trajectory.
    qf : SAVfunction
        Q function.
    lagged_qf : SAVfunction
        Lagged Q function.
    targ_qf1 : CEMSAVfunction
        Target Q function.
    targ_qf2 : CEMSAVfunction
        Lagged Target Q function.
    optim_qf : torch.optim.Optimizer
        Optimizer for Q function.
    epoch : int
        Number of iteration.
    batch_size : int
        Number of batches.
    tau : float
        Target updating rate.
    gamma : float
        Discounting rate.
    loss_type : string
        Type of belleman loss.
    log_enable: bool
        If True, enable logging

    Returns
    -------
    result_dict : dict
        Dictionary which contains losses information.
    """

    qf_losses = []
    if log_enable:
        logger.log("Optimizing...")

    iterator = traj.random_batch(batch_size, epoch)
    for batch in iterator:
        qf_bellman_loss = lf.clipped_double_bellman(
            qf, targ_qf1, targ_qf2, batch, gamma, loss_type=loss_type)
        optim_qf.zero_grad()
        qf_bellman_loss.backward()
        optim_qf.step()

        for q, targ_q1 in zip(qf.parameters(), targ_qf1.parameters()):
            targ_q1.detach().copy_((1 - tau) * targ_q1.detach() + tau * q.detach())

        for lagged_q, targ_q2 in zip(lagged_qf.parameters(), targ_qf2.parameters()):
            targ_q2.detach().copy_((1 - tau) * targ_q2.detach() + tau * lagged_q.detach())

        qf_losses.append(qf_bellman_loss.detach().cpu().numpy())
    if log_enable:
        logger.log("Optimization finished!")
    return {'QfLoss': qf_losses}

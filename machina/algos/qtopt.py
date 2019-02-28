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
          loss_type='mse'
          ):

    qf_losses = []
    logger.log("Optimizing...")

    iterator = traj.random_batch(batch_size, epoch)
    grad_step = len(list(iterator))
    for batch in traj.random_batch(batch_size, epoch):
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
    logger.log("Optimization finished!")
    return {'QfLoss': qf_losses, 'grad_step': grad_step}

"""
These are functions which is applied to trajectory.
"""

import torch
import numpy as np

from machina import loss_functional as lf
from machina.utils import get_device


def update_pris(traj, td_loss, indices, alpha=0.6, epsilon=1e-6, update_epi_pris=False, seq_length=None, eta=0.9):
    """
    Update priorities specified in indices.

    Parameters
    ----------
    data : Traj
    td_loss : torch.Tensor
    indices : torch.Tensor ot List of int
    alpha : float
    epsilon : float
    update_epi_pris : bool
        If True, all priorities of a episode including indices[0] are updated.
    seq_length : int
        Length of batch.
    eta : float

    Returns
    -------
    data : Traj
    """
    pris = (torch.abs(td_loss) + epsilon) ** alpha
    traj.data_map['pris'][indices] = pris.detach().to(traj.traj_device())

    if update_epi_pris:
        epi_start = -1
        epi_end = -1
        seq_start = indices[0]
        for i in range(1, len(traj._epis_index)):
            if seq_start < traj._epis_index[i]:
                epi_start = traj._epis_index[i-1]
                epi_end = traj._epis_index[i]
                break

        pris = traj.data_map['pris'][epi_start: epi_end]
        n_seq = len(pris) - seq_length + 1
        abs_pris = np.abs(pris.cpu().numpy())
        seq_pris = np.array([eta * np.max(abs_pris[i:i+seq_length]) + (1 - eta) *
                             np.mean(abs_pris[i:i+seq_length]) for i in range(n_seq)], dtype='float32')
        traj.data_map['seq_pris'][epi_start:epi_start +
                                  n_seq] = torch.tensor(seq_pris, dtype=torch.float, device=get_device())

    return traj

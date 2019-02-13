"""
These are functions which is applied to trajectory.
"""

import torch

from machina import loss_functional as lf


def update_pris(traj, td_loss, indices, alpha=0.6, epsilon=1e-6):
    pris = (torch.abs(td_loss) + epsilon) ** alpha
    traj.data_map['pris'][indices] = pris.detach()
    return traj

def add_noise_to_obs(traj, std):
    traj.data_map['obs'] = torch.normal(mean=traj.data_map['obs'], std=torch.full_like(traj.data_map['obs'], std))
    traj.data_map['next_obs'] = torch.normal(mean=traj.data_map['next_obs'], std=torch.full_like(traj.data_map['next_obs'], std))
    return traj

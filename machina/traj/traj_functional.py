"""
These are functions which is applied to trajectory.
"""

import torch

from machina import loss_functional as lf


def update_pris(traj, td_loss, indices, alpha=0.6, epsilon=1e-6):
    pris = (torch.abs(td_loss) + epsilon) ** alpha
    traj.data_map['pris'][indices] = pris.detach()
    return traj


def normalize_obs_and_acs(traj):
    with torch.no_grad():
        mean_obs = torch.mean(traj.data_map['obs'], dim=0, keepdim=True)
        std_obs = torch.std(traj.data_map['obs'], dim=0, keepdim=True)
        traj.data_map['obs'] = (traj.data_map['obs'] - mean_obs) / std_obs

        mean_next_obs = torch.mean(
            traj.data_map['next_obs'], dim=0, keepdim=True)
        std_next_obs = torch.std(
            traj.data_map['next_obs'], dim=0, keepdim=True)
        traj.data_map['next_obs'] = (
            traj.data_map['next_obs'] - mean_next_obs) / std_next_obs

        mean_acs = torch.mean(traj.data_map['acs'], dim=0, keepdim=True)
        std_acs = torch.std(traj.data_map['acs'], dim=0, keepdim=True)
        traj.data_map['acs'] = (traj.data_map['acs'] - mean_acs) / std_acs

    return traj, mean_obs, std_obs, mean_next_obs, std_next_obs, mean_acs, std_acs

"""
These are functions which is applied to trajectory.
"""

import torch

from machina import loss_functional as lf


def update_pris(traj, td_loss, indices, alpha=0.6, epsilon=1e-6):
    pris = (torch.abs(td_loss) + epsilon) ** alpha
    traj.data_map['pris'][indices] = pris.detach()
    return traj


def normalize_obs_and_acs(traj, mean_obs=None, std_obs=None, mean_acs=None, std_acs=None, return_statistic=True):
    with torch.no_grad():
        if mean_obs is None:
            mean_obs = torch.mean(traj.data_map['obs'], dim=0, keepdim=True)
        if std_obs is None:
            std_obs = torch.std(traj.data_map['obs'], dim=0, keepdim=True)

        if mean_acs is None:
            mean_acs = torch.mean(traj.data_map['acs'], dim=0, keepdim=True)
        if std_acs is None:
            std_acs = torch.std(traj.data_map['acs'], dim=0, keepdim=True)

        traj.data_map['obs'] = (traj.data_map['obs'] - mean_obs) / std_obs
        traj.data_map['acs'] = (traj.data_map['acs'] - mean_acs) / std_acs
        traj.data_map['next_obs'] = (
            traj.data_map['next_obs'] - mean_obs) / std_obs

        # inf to zero
        obs_inf_mask = traj.data_map['obs'] == float('inf')
        acs_inf_mask = traj.data_map['acs'] == float('inf')
        next_obs_inf_mask = traj.data_map['next_obs'] == float('inf')

        traj.data_map['obs'][obs_inf_mask] = 0
        traj.data_map['acs'][acs_inf_mask] = 0
        traj.data_map['next_obs'][next_obs_inf_mask] = 0

    if return_statistic:
        return traj, mean_obs, std_obs, mean_acs, std_acs
    else:
        return traj

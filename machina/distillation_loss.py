"""
Losses for Policy distillation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from machina.utils import detach_tensor_dict, get_device



def shanon_cross_entropy(student_pol, teacher_pol, batch):
    """
    Shanon-cross entropy as defined in the paper 'Distilling policy distillation'
    https://arxiv.org/abs/1902.02186
    """

    obs = batch['obs']
    acs = batch['acs']

    if teacher_pol.rnn:
        h_masks = batch['h_masks']
        out_masks = batch['out_masks']
    else:
        h_masks = None
        out_masks = torch.ones_like(batch['rews'])

    s_pd = student_pol.pd
    t_pd = teacher_pol.pd

    student_pol.reset()
    teacher_pol.reset()
    _, _, s_params = student_pol(obs, h_masks = h_masks)
    _, _, t_params = teacher_pol(obs, h_masks = h_masks)
    
    s_llh = s_pd(acs, s_params)
    t_llh = t_pd(acsm t_params)
    
    t_lh = torch.exp(t_llh))
    pol_loss = s_llh * t_lh
    pol_loss = torch.mean(pol_loss*out_masks)

    return pol_loss


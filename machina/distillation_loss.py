"""
Losses for Policy distillation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from machina.utils import detach_tensor_dict, get_device


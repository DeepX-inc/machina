import unittest

import torch

from machina.utils import torch2torch


class TestGPU(unittest.TestCase):

    def test_torch2torch(self):
        if not torch.cuda.is_available():
            return
        cuda_tensor = torch2torch(torch.Tensor([0,1,2]))
        self.assertTrue('cuda' in str(type(cuda_tensor)))


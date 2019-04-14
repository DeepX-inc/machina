import os
import unittest

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.nn as nn

from machina.optims import DistributedAdamW


def init_processes(rank, world_size,
                   function, backend='tcp'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank,
                            world_size=world_size)
    function(rank, world_size)


class TestDistributedAdamW(unittest.TestCase):

    def test_step(self):

        def _run(rank, world_size):
            model = nn.Linear(10, 1)
            optimizer = DistributedAdamW(
                model.parameters())

            optimizer.zero_grad()
            loss = model(torch.ones(10).float())
            loss.backward()
            optimizer.step()

        processes = []
        world_size = 4
        for rank in range(world_size):
            p = Process(target=init_processes,
                        args=(rank,
                              world_size,
                              _run))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

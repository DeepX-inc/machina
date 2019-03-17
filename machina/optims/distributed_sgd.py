import torch
from torch.optim import SGD
import torch.distributed as dist


class DistributedSGD(SGD):
    """Distributed SGD optimizer.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        world_size (int, optional): if not given, automatically set
            dist.get_world_size()

    """

    def __init__(self, *args, **kwargs):
        world_size = kwargs.pop('world_size', dist.get_world_size())
        super(DistributedSGD, self).__init__(
            *args, **kwargs)
        self.world_size = world_size

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        world_size = float(self.world_size)
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grads.append(p.grad)
        flat_grads = torch.nn.utils.parameters_to_vector(grads)
        dist.all_reduce_multigpu([flat_grads])
        flat_grads /= world_size
        torch.nn.utils.vector_to_parameters(flat_grads, grads)

        loss = super(DistributedSGD, self).step(closure)
        return loss

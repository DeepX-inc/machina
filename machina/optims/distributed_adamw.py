import math
import torch
from torch.optim.optimizer import Optimizer
import torch.distributed as dist


class DistributedAdamW(Optimizer):
    """Implements AdamW algorithm with distributed settings.

    It has been proposed in `Fixing Weight Decay Regularization in Adam`.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (not L2 penalty) (default: 0)
        local_rank (int, optional): if not given, automatically set
            dist.get_rank()
        world_size (int, optional): if not given, automatically set
            dist.get_world_size()
    """

    def __init__(self, params, local_rank=None, world_size=None,
                 lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(DistributedAdamW, self).__init__(params, defaults)
        if local_rank is None:
            local_rank = dist.get_rank()
        if world_size is None:
            world_size = dist.get_world_size()
        self.local_rank = local_rank
        self.world_size = world_size

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grads.append(p.grad)
        flat_grads = torch.nn.utils.parameters_to_vector(grads)
        dist.all_reduce_multigpu([flat_grads])
        flat_grads /= self.world_size
        torch.nn.utils.vector_to_parameters(flat_grads, grads)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * \
                    math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'], p.data)

                p.data.addcdiv_(-step_size, exp_avg, denom)
        params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)
        params_vec = torch.nn.utils.parameters_to_vector(params)
        dist.broadcast_multigpu([params_vec], 0)
        torch.nn.utils.vector_to_parameters(params_vec, params)

        return loss

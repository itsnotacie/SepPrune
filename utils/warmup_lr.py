import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
import numpy as np

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class CosineAnnealingWithRestartsLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_mult=1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.next_restart = T_max
        self.eta_min = eta_min
        self.restarts = 0
        self.last_restart = 0
        super(
            CosineAnnealingWithRestartsLR,
            self).__init__(
            optimizer,
            last_epoch)

    def get_lr(self):
        self.Tcur = self.last_epoch - self.last_restart
        if self.Tcur >= self.next_restart:
            self.next_restart *= self.T_mult
            self.last_restart = self.last_epoch
        return [(self.eta_min +
                 (base_lr -
                  self.eta_min) *
                 (1 +
                  math.cos(math.pi *
                           self.Tcur /
                           self.next_restart)) /
                 2) for base_lr in self.base_lrs]


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def cosine_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            e = epoch - args.warmup_length
            es = args.epochs - args.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.lr + 0.00001

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length

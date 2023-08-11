"""This module contains an implementation of the LARS optimizer.

Based on the paper:
    "Large Batch Training of Convolutional Networks", https://arxiv.org/abs/1708.03888
"""
import torch
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0005, eta=0.001):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta)
        super(LARS, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                weight = p.data
                weight_norm = torch.norm(weight)
                grad_norm = torch.norm(grad)
                local_lr = (
                    group["eta"]
                    * weight_norm
                    / (grad_norm + group["weight_decay"] * weight_norm)
                )
                velocity = group["momentum"] * grad + group["lr"] * local_lr * grad
                p.data = p.data - velocity

        return loss

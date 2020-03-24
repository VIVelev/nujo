import nujo.objective.functional as F
from nujo.flow import Flow

__all__ = [
    'Loss',
    'MAELoss',
    'MSELoss',
]


class Loss(Flow):
    pass


class MAELoss(Loss):
    def __init__(self, dim=None, keepdim=False):
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input, target):
        return F.mean_abs_err(input,
                              target,
                              dim=self.dim,
                              keepdim=self.keepdim)


class MSELoss(Loss):
    def __init__(self, dim=None, keepdim=False):
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input, target):
        return F.mean_sq_err(input, target, dim=self.dim, keepdim=self.keepdim)

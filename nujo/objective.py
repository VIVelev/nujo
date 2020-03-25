''' More details here: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
'''

from numpy import clip

from nujo.flow import Flow
from nujo.math import abs, log, mean, sum

__all__ = [
    'Loss',
    'L1Loss',
    'L2Loss',
    'BinaryCrossEntropy',
    'CrossEntropy',
]

# ====================================================================================================


class Loss(Flow):
    def __init__(self, dim=None, keepdim=False, reduction='mean'):
        super(Loss, self).__init__(name=self.__class__.__name__)
        self.dim = dim
        self.keepdim = keepdim
        self.reduction = mean if reduction == 'mean' else sum


# ====================================================================================================


class L1Loss(Loss):
    def forward(self, input, target):
        return self.reduction(abs(input - target),
                              dim=self.dim,
                              keepdim=self.keepdim)


# ====================================================================================================


class L2Loss(Loss):
    def forward(self, input, target):
        return self.reduction((input - target)**2,
                              dim=self.dim,
                              keepdim=self.keepdim)


# ====================================================================================================


class BinaryCrossEntropy(Loss):
    def forward(self, input, target):
        # Avoid division by zero
        input.value = clip(input.value, 1e-16, 1 - 1e-16)
        return sum(-target * log(input) - (1 - target) * log(1 - input))


# ====================================================================================================


class CrossEntropy(Loss):
    def forward(self, input, target):
        pass


# ====================================================================================================

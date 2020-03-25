''' More details here: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
'''

from numpy import clip

from nujo.autodiff.tensor import Tensor
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
    def __init__(self, dim: int = None, keepdim=False, reduction: str = None):
        super(Loss, self).__init__(name=self.__class__.__name__)
        self.dim = dim
        self.keepdim = keepdim
        self.reduction = mean if reduction == 'mean' else sum


# ====================================================================================================


class L1Loss(Loss):
    def __init__(self, dim: int = None, keepdim=False, reduction='mean'):
        super(L1Loss, self).__init__(dim, keepdim, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.reduction(abs(input - target),
                              dim=self.dim,
                              keepdim=self.keepdim)


# ====================================================================================================


class L2Loss(Loss):
    def __init__(self, dim: int = None, keepdim=False, reduction='mean'):
        super(L2Loss, self).__init__(dim, keepdim, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.reduction((input - target)**2,
                              dim=self.dim,
                              keepdim=self.keepdim)


# ====================================================================================================


class BinaryCrossEntropy(Loss):
    def __init__(self, dim: int = None, keepdim=False, reduction='sum'):
        super(BinaryCrossEntropy, self).__init__(dim, keepdim, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Avoid division by zero
        input.value = clip(input.value, 1e-16, 1 - 1e-16)
        return self.reduction(-target * log(input) -
                              (1 - target) * log(1 - input),
                              dim=self.dim,
                              keepdim=self.keepdim)


# ====================================================================================================


class CrossEntropy(Loss):
    def __init__(self, dim: int = None, keepdim=False, reduction='sum'):
        super(CrossEntropy, self).__init__(dim, keepdim, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Avoid division by zero
        input.value = clip(input.value, 1e-16, 1 - 1e-16)
        return -self.reduction(sum(target * log(input), dim=1),
                               dim=self.dim,
                               keepdim=self.keepdim)


# ====================================================================================================

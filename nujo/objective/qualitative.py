from numpy import clip

from nujo.autodiff.tensor import Tensor
from nujo.math import log, sum
from nujo.objective.loss import Loss

__all__ = [
    'BinaryCrossEntropy',
    'CrossEntropy',
]

# ====================================================================================================


class BinaryCrossEntropy(Loss):
    ''' Binary Cross-Entropy loss

        −(y * log(p) + (1−y) * log(1 − p))

    '''
    def __init__(self, dim: int = None, keepdim=False, reduction='sum'):
        super(BinaryCrossEntropy, self).__init__(dim, keepdim, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Avoid division by zero
        input.value = clip(input.value, 1e-16, 1 - 1e-16)
        return self.reduction_fn(-target * log(input) -
                                 (1 - target) * log(1 - input),
                                 dim=self.dim,
                                 keepdim=self.keepdim)


# ====================================================================================================


class CrossEntropy(Loss):
    ''' Multi-class Cross-Entropy loss
    '''
    def __init__(self, dim: int = None, keepdim=False, reduction='sum'):
        super(CrossEntropy, self).__init__(dim, keepdim, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Avoid division by zero
        input.value = clip(input.value, 1e-16, 1 - 1e-16)
        return -self.reduction_fn(sum(target * log(input), dim=1),
                                  dim=self.dim,
                                  keepdim=self.keepdim)


# ====================================================================================================

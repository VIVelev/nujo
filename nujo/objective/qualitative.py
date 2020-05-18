from numpy import clip

from nujo.autodiff.tensor import Tensor
from nujo.math import log, sum
from nujo.objective.loss import QualitativeLoss

__all__ = [
    'BinaryCrossEntropy',
    'CrossEntropy',
]

# ====================================================================================================


class BinaryCrossEntropy(QualitativeLoss):
    ''' Binary Cross-Entropy loss

        −(y * log(p) + (1 − y) * log(1 − p))

    '''
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Avoid division by zero
        input.value = clip(input.value, 1e-16, 1 - 1e-16)
        return -self.reduction_fn(target * log(input) +
                                  (1 - target) * log(1 - input),
                                  dim=self.dim,
                                  keepdim=self.keepdim)


# ====================================================================================================


class CrossEntropy(QualitativeLoss):
    ''' Multi-class Cross-Entropy loss

        -∑ y * log(p)

    '''
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Avoid division by zero
        input.value = clip(input.value, 1e-16, 1 - 1e-16)
        return -self.reduction_fn(sum(target * log(input), dim=1,
                                      keepdim=True),
                                  dim=self.dim,
                                  keepdim=self.keepdim)


# ====================================================================================================

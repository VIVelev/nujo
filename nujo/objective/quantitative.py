from nujo.autodiff.tensor import Tensor
from nujo.math.scalar import abs
from nujo.objective.loss import Loss

__all__ = [
    'L1Loss',
    'L2Loss',
]

# ====================================================================================================


class L1Loss(Loss):
    ''' L1 loss (or Mean Absolute Error)
    '''
    def __init__(self, dim: int = None, keepdim=False, reduction='mean'):
        super(L1Loss, self).__init__(dim, keepdim, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.reduction_fn(abs(input - target),
                                 dim=self.dim,
                                 keepdim=self.keepdim)


# ====================================================================================================


class L2Loss(Loss):
    ''' L2 loss (or Mean Squared Error)
    '''
    def __init__(self, dim: int = None, keepdim=False, reduction='mean'):
        super(L2Loss, self).__init__(dim, keepdim, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.reduction_fn((input - target)**2,
                                 dim=self.dim,
                                 keepdim=self.keepdim)


# ====================================================================================================

from nujo.autodiff.tensor import Tensor
from nujo.math.scalar import abs
from nujo.objective.loss import QuantitativeLoss

__all__ = [
    'L1Loss',
    'L2Loss',
]

# ====================================================================================================


class L1Loss(QuantitativeLoss):
    ''' L1 loss (or Absolute Error)

        | ÿ - y |

    '''
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.reduction_fn(abs(input - target),
                                 dim=self.dim,
                                 keepdim=self.keepdim)


# ====================================================================================================


class L2Loss(QuantitativeLoss):
    ''' L2 loss (or Squared Error)

        (ÿ - y)^2

    '''
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.reduction_fn((input - target)**2,
                                 dim=self.dim,
                                 keepdim=self.keepdim)


# ====================================================================================================

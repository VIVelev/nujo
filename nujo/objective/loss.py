''' More details here:
    https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
'''

from nujo.flow import Flow
from nujo.math.aggregate import mean, sum

__all__ = [
    'Loss',
    'QualitativeLoss',
    'QuantitativeLoss',
]

# ====================================================================================================


class Loss(Flow):
    ''' Base loss class
    '''
    def __init__(self, dim: int = None, keepdim=False, reduction: str = None):
        super(Loss, self).__init__(name=self.__class__.__name__)
        self.dim = dim
        self.keepdim = keepdim

        if reduction == 'sum':
            self.reduction_fn = sum
        elif reduction == 'mean':
            self.reduction_fn = mean
        else:
            self.reduction_fn = lambda x: x


# ====================================================================================================


class QualitativeLoss(Loss):
    ''' Base qualitative loss class
    '''
    def __init__(self, dim: int = None, keepdim=False, reduction='sum'):
        super(QualitativeLoss, self).__init__(dim, keepdim, reduction)


# ====================================================================================================


class QuantitativeLoss(Loss):
    ''' Base quantitative loss class
    '''
    def __init__(self, dim: int = None, keepdim=False, reduction='mean'):
        super(QuantitativeLoss, self).__init__(dim, keepdim, reduction)


# ====================================================================================================

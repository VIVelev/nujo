from nujo.flow import Flow
from nujo.math.aggregate import mean, sum

__all__ = [
    'QualitativeLoss',
    'QuantitativeLoss',
]

# ====================================================================================================


class _Loss(Flow):
    ''' Base loss class
    '''
    def __init__(self, dim: int = None, keepdim=False, reduction: str = None):
        super(_Loss, self).__init__(name=self.__class__.__name__)
        self.dim = dim
        self.keepdim = keepdim

        if reduction == 'sum':
            self.reduction_fn = sum
        elif reduction == 'mean':
            self.reduction_fn = mean
        else:  # if None
            self.reduction_fn = lambda x: x


# ====================================================================================================


class QualitativeLoss(_Loss):
    ''' Base qualitative loss class
    '''
    def __init__(self, dim: int = None, keepdim=False, reduction='sum'):
        super(QualitativeLoss, self).__init__(dim, keepdim, reduction)


# ====================================================================================================


class QuantitativeLoss(_Loss):
    ''' Base quantitative loss class
    '''
    def __init__(self, dim: int = None, keepdim=False, reduction='mean'):
        super(QuantitativeLoss, self).__init__(dim, keepdim, reduction)


# ====================================================================================================

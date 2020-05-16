from typing import Optional

from nujo.flow import Flow
from nujo.math.aggregate import mean, sum

__all__ = [
    'QualitativeLoss',
    'QuantitativeLoss',
]

# ====================================================================================================


class _Loss(Flow):
    ''' Base Loss Function Class

    Do NOT inherit this class directly. Instead, inherit either
    `QualitativeLoss` or `QuantitativeLoss`, depending on the task
    for which you implement the loss function (classification/regression).

    Parameters:
    -----------
     - dim : int (optional), the dimension along which to reduce
     - keepdim : bool, whether to keep the dimension
     - reduction, string (optional), reduction function ('sum', 'mean', etc.)

    '''
    def __init__(self,
                 dim: Optional[int] = None,
                 keepdim=True,
                 reduction: Optional[str] = None):

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
    ''' Base Qualitative (Classification) Loss Function Class

    If you want to implement a custom loss function for classification,
    inherit this class.

    Parameters:
    -----------
     - dim : int (optional), the dimension along which to reduce
     - keepdim : bool, whether to keep the dimension
     - reduction, string (optional), reduction function (default: 'sum')

    '''
    def __init__(self,
                 dim: Optional[int] = None,
                 keepdim=True,
                 reduction='sum'):

        super(QualitativeLoss, self).__init__(dim, keepdim, reduction)


# ====================================================================================================


class QuantitativeLoss(_Loss):
    ''' Base Quantitative (Regression) Loss Function Class

    If you want to implement a custom loss function for regression,
    inherit this class.

    Parameters:
    -----------
     - dim : int (optional), the dimension along which to reduce
     - keepdim : bool, whether to keep the dimension
     - reduction, string (optional), reduction function (default: 'mean')

    '''
    def __init__(self,
                 dim: Optional[int] = None,
                 keepdim=True,
                 reduction='mean'):

        super(QuantitativeLoss, self).__init__(dim, keepdim, reduction)


# ====================================================================================================

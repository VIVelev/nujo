from nujo.autodiff._functions._aggregate import _InnerProd, _InnerSum
from nujo.autodiff.tensor import Tensor
from nujo.flow import Flow

__all__ = [
    'sum',
    'prod',
    'mean',
]

# ====================================================================================================


class sum(Flow):
    ''' Summation of tensors

    Parameters:
    -----------
    args : varargs, tensors to be summed;
    if a single tensor is passed, its elements will be summed
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`

    Returns:
    --------
    result : Tensor

    '''
    def __init__(self, dim=0, keepdim=False):
        super(sum, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, *inputs: Tensor) -> Tensor:
        if len(inputs) == 1:
            return _InnerSum(inputs[0], dim=self.dim, keepdim=self.keepdim)()
        else:
            pass


# ====================================================================================================


class prod(Flow):
    ''' Product of tensors

    Parameters:
    -----------
    args : varargs, tensors to be multiplied;
    if a single tensor is passed, its elements will be multiplied
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`

    Returns:
    --------
    result : Tensor

    '''
    def __init__(self, dim=0, keepdim=False):
        super(prod, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, *inputs: Tensor) -> Tensor:
        if len(inputs) == 1:
            return _InnerProd(inputs[0], dim=self.dim, keepdim=self.keepdim)()
        else:
            pass


# ====================================================================================================


class mean(Flow):
    ''' Mean of tensors

    Parameters:
    -----------
    args : varargs, tensors to compute the mean of;
    if a single tensor is passed, the mean of its elements will be computed
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`

    Returns:
    --------
    result : Tensor

    '''
    def __init__(self, dim=0, keepdim=False):
        super(mean, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, *inputs: Tensor) -> Tensor:
        if len(inputs) == 1:
            return _InnerSum(
                inputs[0], dim=self.dim,
                keepdim=self.keepdim)() / inputs[0].shape[self.dim]
        else:
            pass


# ====================================================================================================

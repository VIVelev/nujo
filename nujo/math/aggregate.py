from typing import Optional

from numpy import prod as np_prod
from numpy import sum as np_sum

from nujo.autodiff._functions._aggregate import _InnerProd, _InnerSum
from nujo.autodiff.tensor import Tensor

__all__ = [
    'sum',
    'prod',
    'mean',
]

# ====================================================================================================


def sum(*inputs: Tensor, dim: Optional[int] = None, keepdim=False) -> Tensor:
    ''' Summation of tensor(s)

    Parameters:
    -----------
     - inputs : varargs, tensors to be summed;
       if a single tensor is passed, its elements will be summed
     - dim : int (optional), dimension to reduce over
     - keepdim : bool, whether to keep `dim`

    Returns:
    --------
     - result : Tensor

    '''

    if len(inputs) == 1:
        return _InnerSum(inputs[0], dim=dim, keepdim=keepdim)()
    else:
        return np_sum(inputs, axis=dim, keepdims=keepdim)


# ====================================================================================================


def prod(*inputs: Tensor, dim: Optional[int] = None, keepdim=False) -> Tensor:
    ''' Product of tensor(s)

    Parameters:
    -----------
     - inputs : varargs, tensors to be multiplied;
       if a single tensor is passed, its elements will be multiplied
     - dim : int (optional), dimension to reduce over
     - keepdim : bool, whether to keep `dim`

    Returns:
    --------
     - result : Tensor

    '''

    if len(inputs) == 1:
        return _InnerProd(inputs[0], dim=dim, keepdim=keepdim)()
    else:
        return np_prod(inputs, axis=dim, keepdims=keepdim)


# ====================================================================================================


def mean(*inputs: Tensor, dim: Optional[int] = None, keepdim=False) -> Tensor:
    ''' Mean of tensor(s)

    Parameters:
    -----------
     - inputs : varargs, tensors to compute the mean of;
       if a single tensor is passed, the mean of its elements will be computed
     - dim : int (optional), dimension to reduce over
     - keepdim : bool, whether to keep `dim`

    Returns:
    --------
     - result : Tensor

    '''

    if len(inputs) == 1:
        n = np_prod(inputs[0].shape) if dim is None else inputs[0].shape[dim]
        return _InnerSum(inputs[0], dim=dim, keepdim=keepdim)() / n
    else:
        return np_sum(inputs, axis=dim, keepdims=keepdim) / len(inputs)


# ====================================================================================================

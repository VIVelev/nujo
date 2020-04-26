from numbers import Number

from numpy import max as np_max
from numpy import mean as np_mean
from numpy import median as np_median
from numpy import min as np_min
from numpy import prod as np_prod
from numpy import std as np_stddev
from numpy import sum as np_sum
from numpy import var as np_variance

from nujo.autodiff.tensor import Tensor

__all__ = [
    'sum',
    'prod',
    'mean',
    'median',
    'min',
    'max',
    'stddev',
    'variance',
]

# ====================================================================================================


def sum(*args: Number or Tensor, dim: int = None, keepdim=False) -> Tensor:
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

    args = [arg if isinstance(arg, Tensor) else Tensor(arg) for arg in args]

    if len(args) > 1:
        return np_sum(args, axis=dim, keepdims=keepdim)

    else:
        args[0].value = np_sum(args[0].value, axis=dim, keepdims=keepdim)
        return args[0]


# ====================================================================================================


def prod(*args: Number or Tensor, dim: int = None, keepdim=False) -> Tensor:
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

    args = [arg if isinstance(arg, Tensor) else Tensor(arg) for arg in args]

    if len(args) > 1:
        return np_prod(args, axis=dim, keepdims=keepdim)

    else:
        args[0].value = np_prod(args[0].value, axis=dim, keepdims=keepdim)
        return args[0]


# ====================================================================================================


def mean(*args: Number or Tensor, dim: int = None, keepdim=False) -> Tensor:
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

    args = [arg if isinstance(arg, Tensor) else Tensor(arg) for arg in args]

    if len(args) > 1:
        return np_mean(args, axis=dim, keepdims=keepdim)

    else:
        args[0].value = np_mean(args[0].value, axis=dim, keepdims=keepdim)
        return args[0]


# ====================================================================================================


def median(*args: Number or Tensor, dim: int = None, keepdim=False) -> Tensor:
    ''' Median of tensors

    Parameters:
    -----------
    args : varargs, tensors to compute the median of;
    if a single tensor is passed, the median of its elements will be computed
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`

    Returns:
    --------
    result : Tensor

    '''

    args = [arg if isinstance(arg, Tensor) else Tensor(arg) for arg in args]

    if len(args) > 1:
        return np_median(args, axis=dim, keepdims=keepdim)

    else:
        args[0].value = np_median(args[0].value, axis=dim, keepdims=keepdim)
        return args[0]


# ====================================================================================================


def min(*args: Number or Tensor, dim: int = None, keepdim=False) -> Tensor:
    ''' Min of tensors

    Parameters:
    -----------
    args : varargs, tensors to compute the min of;
    if a single tensor is passed, the min of its elements will be computed
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`

    Returns:
    --------
    result : Tensor

    '''

    args = [arg if isinstance(arg, Tensor) else Tensor(arg) for arg in args]

    if len(args) > 1:
        return np_min(args, axis=dim, keepdims=keepdim)

    else:
        args[0].value = np_min(args[0].value, axis=dim, keepdims=keepdim)
        return args[0]


# ====================================================================================================


def max(*args: Number or Tensor, dim: int = None, keepdim=False) -> Tensor:
    ''' Max of tensors

    Parameters:
    -----------
    args : varargs, tensors to compute the max of;
    if a single tensor is passed, the max of its elements will be computed
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`

    Returns:
    --------
    result : Tensor

    '''

    args = [arg if isinstance(arg, Tensor) else Tensor(arg) for arg in args]

    if len(args) > 1:
        return np_max(args, axis=dim, keepdims=keepdim)

    else:
        args[0].value = np_max(args[0].value, axis=dim, keepdims=keepdim)
        return args[0]


# ====================================================================================================


def stddev(*args: Number or Tensor, dim: int = None, keepdim=False) -> Tensor:
    ''' Standard Deviation of tensors

    Parameters:
    -----------
    args : varargs, tensors to compute the std dev of;
    if a single tensor is passed, the std dev of its elements will be computed
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`

    Returns:
    --------
    result : Tensor

    '''

    args = [arg if isinstance(arg, Tensor) else Tensor(arg) for arg in args]

    if len(args) > 1:
        return np_stddev(args, axis=dim, keepdims=keepdim)

    else:
        args[0].value = np_stddev(args[0].value, axis=dim, keepdims=keepdim)
        return args[0]


# ====================================================================================================


def variance(*args: Number or Tensor,
             dim: int = None,
             keepdim=False) -> Tensor:
    ''' Variance of tensors

    Parameters:
    -----------
    args : varargs, tensors to compute the variance of;
    if a single tensor is passed, the variance of its elements will be computed
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`

    Returns:
    --------
    result : Tensor

    '''

    args = [arg if isinstance(arg, Tensor) else Tensor(arg) for arg in args]

    if len(args) > 1:
        return np_variance(args, axis=dim, keepdims=keepdim)

    else:
        args[0].value = np_variance(args[0].value, axis=dim, keepdims=keepdim)
        return args[0]


# ====================================================================================================

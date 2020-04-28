from copy import deepcopy

from numpy import max as np_max
from numpy import mean as np_mean
from numpy import median as np_median
from numpy import min as np_min
from numpy import prod as np_prod
from numpy import std as np_stddev
from numpy import sum as np_sum
from numpy import var as np_variance

from nujo._typing import Union, _numerical
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


def sum(*args: Union[Tensor, _numerical],
        dim: int = None,
        keepdim=False,
        inplace=False) -> Tensor:
    ''' Summation of tensors

    Parameters:
    -----------
    args : varargs, tensors to be summed;
    if a single tensor is passed, its elements will be summed
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`
    inplace : bool, whether to make the computation in-place;
    a.k.a. take the argument by reference instead of by value;
    (this parameter is taken into account only if a single argument is passed)

    Returns:
    --------
    result : Tensor

    '''

    tensor_0 = args[0] if isinstance(args[0], Tensor) else Tensor(args[0])

    if len(args) > 1:
        res_value = np_sum(
            [arg.value if isinstance(arg, Tensor) else arg for arg in args],
            axis=dim,
            keepdims=keepdim)

    else:
        res_value = np_sum(tensor_0.value, axis=dim, keepdims=keepdim)

    res = tensor_0 if inplace else deepcopy(tensor_0)
    res.name += ' (sum)'
    res.value = res_value

    return res


# ====================================================================================================


def prod(*args: Union[Tensor, _numerical],
         dim: int = None,
         keepdim=False,
         inplace=False) -> Tensor:
    ''' Product of tensors

    Parameters:
    -----------
    args : varargs, tensors to be multiplied;
    if a single tensor is passed, its elements will be multiplied
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`
    inplace : bool, whether to make the computation in-place;
    a.k.a. take the argument by reference instead of by value;
    (this parameter is taken into account only if a single argument is passed)

    Returns:
    --------
    result : Tensor

    '''

    tensor_0 = args[0] if isinstance(args[0], Tensor) else Tensor(args[0])

    if len(args) > 1:
        res_value = np_prod(
            [arg.value if isinstance(arg, Tensor) else arg for arg in args],
            axis=dim,
            keepdims=keepdim)

    else:
        res_value = np_prod(tensor_0.value, axis=dim, keepdims=keepdim)

    res = tensor_0 if inplace else deepcopy(tensor_0)
    res.name += ' (prod)'
    res.value = res_value

    return res


# ====================================================================================================


def mean(*args: Union[Tensor, _numerical],
         dim: int = None,
         keepdim=False,
         inplace=False) -> Tensor:
    ''' Mean of tensors

    Parameters:
    -----------
    args : varargs, tensors to compute the mean of;
    if a single tensor is passed, the mean of its elements will be computed
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`
    inplace : bool, whether to make the computation in-place;
    a.k.a. take the argument by reference instead of by value;
    (this parameter is taken into account only if a single argument is passed)

    Returns:
    --------
    result : Tensor

    '''

    tensor_0 = args[0] if isinstance(args[0], Tensor) else Tensor(args[0])

    if len(args) > 1:
        res_value = np_mean(
            [arg.value if isinstance(arg, Tensor) else arg for arg in args],
            axis=dim,
            keepdims=keepdim)

    else:
        res_value = np_mean(tensor_0.value, axis=dim, keepdims=keepdim)

    res = tensor_0 if inplace else deepcopy(tensor_0)
    res.name += ' (mean)'
    res.value = res_value

    return res


# ====================================================================================================


def median(*args: Union[Tensor, _numerical],
           dim: int = None,
           keepdim=False,
           inplace=False) -> Tensor:
    ''' Median of tensors

    Parameters:
    -----------
    args : varargs, tensors to compute the median of;
    if a single tensor is passed, the median of its elements will be computed
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`
    inplace : bool, whether to make the computation in-place;
    a.k.a. take the argument by reference instead of by value;
    (this parameter is taken into account only if a single argument is passed)

    Returns:
    --------
    result : Tensor

    '''

    tensor_0 = args[0] if isinstance(args[0], Tensor) else Tensor(args[0])

    if len(args) > 1:
        res_value = np_median(
            [arg.value if isinstance(arg, Tensor) else arg for arg in args],
            axis=dim,
            keepdims=keepdim)

    else:
        res_value = np_median(tensor_0.value, axis=dim, keepdims=keepdim)

    res = tensor_0 if inplace else deepcopy(tensor_0)
    res.name += ' (median)'
    res.value = res_value

    return res


# ====================================================================================================


def min(*args: Union[Tensor, _numerical],
        dim: int = None,
        keepdim=False,
        inplace=False) -> Tensor:
    ''' Min of tensors

    Parameters:
    -----------
    args : varargs, tensors to compute the min of;
    if a single tensor is passed, the min of its elements will be computed
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`
    inplace : bool, whether to make the computation in-place;
    a.k.a. take the argument by reference instead of by value;
    (this parameter is taken into account only if a single argument is passed)

    Returns:
    --------
    result : Tensor

    '''

    tensor_0 = args[0] if isinstance(args[0], Tensor) else Tensor(args[0])

    if len(args) > 1:
        res_value = np_min(
            [arg.value if isinstance(arg, Tensor) else arg for arg in args],
            axis=dim,
            keepdims=keepdim)

    else:
        res_value = np_min(tensor_0.value, axis=dim, keepdims=keepdim)

    res = tensor_0 if inplace else deepcopy(tensor_0)
    res.name += ' (min)'
    res.value = res_value

    return res


# ====================================================================================================


def max(*args: Union[Tensor, _numerical],
        dim: int = None,
        keepdim=False,
        inplace=False) -> Tensor:
    ''' Max of tensors

    Parameters:
    -----------
    args : varargs, tensors to compute the max of;
    if a single tensor is passed, the max of its elements will be computed
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`
    inplace : bool, whether to make the computation in-place;
    a.k.a. take the argument by reference instead of by value;
    (this parameter is taken into account only if a single argument is passed)

    Returns:
    --------
    result : Tensor

    '''

    tensor_0 = args[0] if isinstance(args[0], Tensor) else Tensor(args[0])

    if len(args) > 1:
        res_value = np_max(
            [arg.value if isinstance(arg, Tensor) else arg for arg in args],
            axis=dim,
            keepdims=keepdim)

    else:
        res_value = np_max(tensor_0.value, axis=dim, keepdims=keepdim)

    res = tensor_0 if inplace else deepcopy(tensor_0)
    res.name += ' (max)'
    res.value = res_value

    return res


# ====================================================================================================


def stddev(*args: Union[Tensor, _numerical],
           dim: int = None,
           keepdim=False,
           inplace=False) -> Tensor:
    ''' Standard Deviation of tensors

    Parameters:
    -----------
    args : varargs, tensors to compute the std dev of;
    if a single tensor is passed, the std dev of its elements will be computed
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`
    inplace : bool, whether to make the computation in-place;
    a.k.a. take the argument by reference instead of by value;
    (this parameter is taken into account only if a single argument is passed)

    Returns:
    --------
    result : Tensor

    '''

    tensor_0 = args[0] if isinstance(args[0], Tensor) else Tensor(args[0])

    if len(args) > 1:
        res_value = np_stddev(
            [arg.value if isinstance(arg, Tensor) else arg for arg in args],
            axis=dim,
            keepdims=keepdim)

    else:
        res_value = np_stddev(tensor_0.value, axis=dim, keepdims=keepdim)

    res = tensor_0 if inplace else deepcopy(tensor_0)
    res.name += ' (stddev)'
    res.value = res_value

    return res


# ====================================================================================================


def variance(*args: Union[Tensor, _numerical],
             dim: int = None,
             keepdim=False,
             inplace=False) -> Tensor:
    ''' Variance of tensors

    Parameters:
    -----------
    args : varargs, tensors to compute the variance of;
    if a single tensor is passed, the variance of its elements will be computed
    dim : int, dimension to reduce over
    keepdim : bool, whether to keep `dim`
    inplace : bool, whether to make the computation in-place;
    a.k.a. take the argument by reference instead of by value;
    (this parameter is taken into account only if a single argument is passed)

    Returns:
    --------
    result : Tensor

    '''

    tensor_0 = args[0] if isinstance(args[0], Tensor) else Tensor(args[0])

    if len(args) > 1:
        res_value = np_variance(
            [arg.value if isinstance(arg, Tensor) else arg for arg in args],
            axis=dim,
            keepdims=keepdim)

    else:
        res_value = np_variance(tensor_0.value, axis=dim, keepdims=keepdim)

    res = tensor_0 if inplace else deepcopy(tensor_0)
    res.name += ' (variance)'
    res.value = res_value

    return res


# ====================================================================================================

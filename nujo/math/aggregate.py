from copy import deepcopy

from nujo._typing import Union, _numerical
from nujo.autodiff._functions import _InnerMean
from nujo.autodiff.tensor import Tensor
from nujo.flow import Flow

__all__ = [
    'sum',
    'prod',
    'mean',
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


class mean(Flow):
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
    def __init__(self, dim: int = None, keepdim=False, inplace=False):
        self.dim = dim
        self.keepdim = keepdim
        self.inplace = inplace

    def forward(self, *inputs: Tensor) -> Tensor:
        if len(inputs) == 1:
            return _InnerMean(inputs[0], dim=self.dim, keepdim=self.keepdim)()
        else:
            return sum(inputs) / len(inputs)


# ====================================================================================================

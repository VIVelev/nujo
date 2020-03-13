from numpy import prod as np_prod
from numpy import sum as np_sum

from nujo.autodiff import Tensor

__all__ = [
    'sum',
    'prod',
]

# ====================================================================================================


def sum(*args: Tensor, dim: int = None, keepdim=False) -> Tensor:
    ''' Summation of tensors

    Parameters:
    -----------
    args : varargs, tensors to be summed;
    if a single tensor is passed, its elements will be summed
    dim : int, dimensional to reduce
    keepdim : bool, whether to keep `dim`

    Returns:
    --------
    result : Tensor

    '''

    if len(args) > 1:
        return np_sum(args, axis=dim, keepdims=keepdim)

    else:
        return Tensor(np_sum(args[0].value, axis=dim, keepdims=keepdim),
                      creator=args[0].creator)


# ====================================================================================================


def prod(*args: Tensor, dim: int = None, keepdim=False) -> Tensor:
    ''' Product of tensors

    Parameters:
    -----------
    args : varargs, tensors to be multiplied;
    if a single tensor is passed, its elements will be multiplied
    dim : int, dimensional to reduce
    keepdim : bool, whether to keep `dim`

    Returns:
    --------
    result : Tensor

    '''

    if len(args) > 1:
        return np_prod(args, axis=dim, keepdims=keepdim)

    else:
        return Tensor(np_prod(args[0].value, axis=dim, keepdims=keepdim),
                      creator=args[0].creator)


# ====================================================================================================

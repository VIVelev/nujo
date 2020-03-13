from numpy import max as np_max
from numpy import mean as np_mean
from numpy import median as np_median
from numpy import min as np_min

from nujo.autodiff import Tensor

__all__ = [
    'nj_mean',
    'nj_median',
    'nj_min',
    'nj_max',
]

# TODO:
# Add __doc__
# implement some more functions
# is `nj_` prefix necessary ?
# add reshape op to Tensor

# ====================================================================================================


def nj_mean(*args: Tensor, dim: int = None, keepdim=False) -> Tensor:
    ''' Mean of tensors

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
        return np_mean(args, axis=dim, keepdims=keepdim)

    else:
        return Tensor(np_mean(args[0].value, axis=dim, keepdims=keepdim),
                      creator=args[0].creator)


# ====================================================================================================


def nj_median(*args: Tensor, dim: int = None, keepdim=False) -> Tensor:
    ''' Median of tensors

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
        return np_median(args, axis=dim, keepdims=keepdim)

    else:
        return Tensor(np_median(args[0].value, axis=dim, keepdims=keepdim),
                      creator=args[0].creator)


# ====================================================================================================


def nj_min(*args: Tensor, dim: int = None, keepdim=False) -> Tensor:
    ''' Min of tensors

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
        return np_min(args, axis=dim, keepdims=keepdim)

    else:
        return Tensor(np_min(args[0].value, axis=dim, keepdims=keepdim),
                      creator=args[0].creator)


# ====================================================================================================


def nj_max(*args: Tensor, dim: int = None, keepdim=False) -> Tensor:
    ''' Max of tensors

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
        return np_max(args, axis=dim, keepdims=keepdim)

    else:
        return Tensor(np_max(args[0].value, axis=dim, keepdims=keepdim),
                      creator=args[0].creator)


# ====================================================================================================

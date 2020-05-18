from numpy import empty as np_empty
from numpy import full as np_full

from nujo.autodiff.tensor import Tensor

__all__ = [
    'empty',
    'full',
    'ones',
    'ones_like',
    'zeros',
    'zeros_like',
]

# ====================================================================================================


def empty(*shape: int, diff=False, name='Tensor[empty]') -> Tensor:
    ''' Return a new Tensor of given shape, without initializing entries.
    '''

    return Tensor(np_empty(shape), diff=diff, name=name)


def full(*shape: int,
         fill_value=0,
         diff=False,
         name='Tensor[full]]') -> Tensor:
    ''' Return a new Tensor of given shape, filled with `fill_value`.
    '''

    return Tensor(np_full(shape, fill_value), diff=diff, name=name)


# ====================================================================================================


def ones(*shape: int, diff=False, name='Tensor[ones]') -> Tensor:
    ''' Return a new Tensor of given shape, filled with ones.
    '''

    return full(*shape, fill_value=1, diff=diff, name=name)


def ones_like(x: Tensor, diff=False, name='Tensor[ones]') -> Tensor:
    return ones(*x.shape, diff=diff, name=name)


# ====================================================================================================


def zeros(*shape: int, diff=False, name='Tensor[zeros]') -> Tensor:
    ''' Return a new Tensor of given shape, filled with zeros.
    '''

    return full(*shape, fill_value=0, diff=diff, name=name)


def zeros_like(x: Tensor, diff=False, name='Tensor[zeros]') -> Tensor:
    return zeros(*x.shape, diff=diff, name=name)


# ====================================================================================================

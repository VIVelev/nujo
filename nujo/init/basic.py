from numpy import empty as np_empty
from numpy import full as np_full
from numpy import ones as np_ones
from numpy import zeros as np_zeros

from nujo.autodiff.tensor import Tensor

__all__ = [
    'empty',
    'full',
    'ones',
    'ones_like',
    'zeros',
    'zeros_like',
]


def empty(*shape: int, diff=True, name='Tensor[empty]') -> Tensor:
    ''' Return a new array of given shape, without initializing entries.
    '''

    return Tensor(np_empty(shape), diff=diff, name=name)


def full(*shape: int, fill_value=0, diff=True, name='Tensor[full]]') -> Tensor:
    ''' Return a new array of given shape, filled with `fill_value`.
    '''

    return Tensor(np_full(shape, fill_value), diff=diff, name=name)


def ones(*shape: int, diff=True, name='Tensor[ones]') -> Tensor:
    ''' Return a new array of given shape, filled with ones.
    '''

    return Tensor(np_ones(shape), diff=diff, name=name)


def ones_like(x: Tensor, diff=True, name='Tensor[ones]') -> Tensor:
    return ones(*x.shape, diff=diff, name=name)


def zeros(*shape: int, diff=True, name='Tensor[zeros]') -> Tensor:
    ''' Return a new array of given shape, filled with zeros.
    '''

    return Tensor(np_zeros(shape), diff=diff, name=name)


def zeros_like(x: Tensor, diff=True, name='Tensor[zeros]') -> Tensor:
    return zeros(*x.shape, diff=diff, name=name)

from numpy import empty as np_empty
from numpy import ones as np_ones
from numpy import zeros as np_zeros

from nujo.autodiff.tensor import Tensor

__all__ = [
    'empty',
    'ones',
    'zeros',
]


def empty(shape: tuple, diff=True, name='<empty::Tensor>') -> Tensor:
    return Tensor(np_empty(shape), diff=diff, name=name)


def ones(shape: tuple, diff=True, name='<ones::Tensor>') -> Tensor:
    return Tensor(np_ones(shape), diff=diff, name=name)


def zeros(shape: tuple, diff=True, name='<zeros::Tensor>') -> Tensor:
    return Tensor(np_zeros(shape), diff=diff, name=name)

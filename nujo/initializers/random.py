from numpy.random import rand as np_rand
from numpy.random import randint as np_randint
from numpy.random import randn as np_randn

from nujo.autodiff.tensor import Tensor

__all__ = [
    'rand',
    'randn',
    'randint',
]


def rand(*shape: int, diff=True, name='<rand::Tensor>') -> Tensor:
    return Tensor(np_rand(*shape), diff=diff, name=name)


def randn(*shape: int, diff=True, name='<randn::Tensor>') -> Tensor:
    return Tensor(np_randn(*shape), diff=diff, name=name)


def randint(*shape: int,
            low=0,
            high=100,
            diff=True,
            name='<randint::Tensor>') -> Tensor:

    return Tensor(np_randint(low, high=high, size=shape), diff=diff, name=name)

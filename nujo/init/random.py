from numpy.random import rand as np_rand
from numpy.random import randint as np_randint
from numpy.random import randn as np_randn

from nujo.autodiff.tensor import Tensor

__all__ = [
    'rand',
    'randn',
    'randint',
]


def rand(*shape: int, diff=False, name='Tensor[rand]') -> Tensor:
    ''' Random values in a given shape.
    '''

    return Tensor(np_rand(*shape), diff=diff, name=name)


def randn(*shape: int, diff=False, name='Tensor[randn]') -> Tensor:
    ''' Return a sample (or samples) from the "standard normal" distribution.
    '''

    return Tensor(np_randn(*shape), diff=diff, name=name)


def randint(*shape: int,
            low=0,
            high=100,
            diff=False,
            name='Tensor[randint]') -> Tensor:
    ''' Return random integers from low (inclusive) to high (exclusive).
    '''

    return Tensor(np_randint(low, high=high, size=shape), diff=diff, name=name)

from numbers import Number
from typing import List, Union

from numpy import ndarray

__all__ = [
    'Union',
    '_array',
    '_numerical',
]

_array = Union[ndarray, List[Number]]
_numerical = Union[_array, Number]

from numpy import prod as np_prod
from numpy import sum as np_sum

from nujo.autodiff import Tensor
from nujo.autodiff.functions import (Addition, MatrixMul, Multiplication,
                                     Negation, Power, Reciprocal)

__all__ = [
    'add',
    'neg',
    'mul',
    'recipr',
    'pow',
    'matmul',
    'sum',
    'prod',
]

# ====================================================================================================


def add(input_a: Tensor, input_b: Tensor) -> Tensor:
    return Addition(input_a, input_b)()


def neg(input: Tensor) -> Tensor:
    return Negation(input)()


def mul(input_a: Tensor, input_b: Tensor) -> Tensor:
    return Multiplication(input_a, input_b)()


def recipr(input: Tensor) -> Tensor:
    return Reciprocal(input)()


def pow(input_a: Tensor, input_b: Tensor) -> Tensor:
    return Power(input_a, input_b)()


def matmul(input_a: Tensor, input_b: Tensor) -> Tensor:
    return MatrixMul(input_a, input_b)()


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

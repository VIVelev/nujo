from numbers import Number
from typing import List, Union

from numpy import log, ndarray, ones

from nujo.autodiff.function import Function
from nujo.autodiff.tensor import Tensor

__all__ = [
    '_Addition',
    '_Negation',
    '_Multiplication',
    '_Reciprocal',
    '_Power',
    '_Logarithm',
    '_MatrixMul',
]

# ====================================================================================================


class _Addition(Function):
    def __init__(self, input_a: Union[Tensor, ndarray, List[Number], Number],
                 input_b: Union[Tensor, ndarray, List[Number], Number]):

        super(_Addition, self).__init__(input_a, input_b)

        # The following assert will not allow numpy's
        # vector broadcasts such as:
        #
        #   [[1, 2, 3]] + [[1], = [[2, 3, 4],
        #                  [2],    [3, 4, 5],
        #                  [3]]    [4, 5, 6]]
        #
        # In future versions of nujo this may be supported.

        assert (self.children[0].value.shape == self.children[1].value.shape or
                self.children[0].value.shape != self.children[1].value.T.shape)

    def forward(self) -> ndarray:
        return self.children[0].value + self.children[1].value

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad * ones(self.children[idx].shape)


# ====================================================================================================


class _Negation(Function):
    def __init__(self, input: Union[Tensor, ndarray, List[Number], Number]):
        super(_Negation, self).__init__(input)

    def forward(self) -> ndarray:
        return -self.children[0].value

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad * -ones(self.children[0].shape)


# ====================================================================================================


class _Multiplication(Function):
    def __init__(self, input_a: Union[Tensor, ndarray, List[Number], Number],
                 input_b: Union[Tensor, ndarray, List[Number], Number]):

        super(_Multiplication, self).__init__(input_a, input_b)

        # The following assert will not allow numpy's
        # vector broadcasts such as:
        #
        #   [[1, 2, 3]] * [[1], = [[1, 2, 3],
        #                  [2],    [2, 4, 6],
        #                  [3]]    [3, 6, 6]]
        #
        # In future versions of nujo this may be supported.

        assert (self.children[0].value.shape == self.children[1].value.shape or
                self.children[0].value.shape != self.children[1].value.T.shape)

    def forward(self) -> ndarray:
        return self.children[0].value * self.children[1].value

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        if idx == 0:
            return accum_grad * self.children[1].value
        else:
            return accum_grad * self.children[0].value


# ====================================================================================================


class _Reciprocal(Function):
    def __init__(self,
                 input: Union[Tensor, ndarray, List[Number], Number],
                 eps=1e-18):

        super(_Reciprocal, self).__init__(input)
        self.eps = eps

    def forward(self) -> ndarray:
        return 1 / (self.children[0].value + self.eps)

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad * -1 / ((self.children[0].value + self.eps)**2)


# ====================================================================================================


class _Power(Function):
    def __init__(self, input_a: Union[Tensor, ndarray, List[Number], Number],
                 input_b: Union[Tensor, ndarray, List[Number], Number]):

        super(_Power, self).__init__(input_a, input_b)

    def forward(self) -> ndarray:
        return self.children[0].value**self.children[1].value

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        # TODO: FIX wrong partial - the second

        if idx == 0:
            return accum_grad * self.children[1].value *\
                    self.children[0].value**(self.children[1].value - 1)
        else:
            return type(accum_grad)(1)


# ====================================================================================================


class _Logarithm(Function):
    def __init__(self, input_a: Union[Tensor, ndarray, List[Number], Number],
                 input_b: Union[Tensor, ndarray, List[Number], Number]):

        super(_Logarithm, self).__init__(input_a, input_b)

        assert (self.children[0].value > 0).all()  # argument value limit
        assert (self.children[1].value > 0).all()  # base value limit
        assert (self.children[1].value != 0).all()  # base value limit

    def forward(self) -> ndarray:
        return log(self.children[0].value) / log(self.children[1].value)

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        # TODO: FIX wrong partial - the second

        if idx == 0:
            return accum_grad /\
                    (self.children[0].value * log(self.children[1].value))
        else:
            return type(accum_grad)(1)


# ====================================================================================================


class _MatrixMul(Function):
    def __init__(self, input_a: Union[Tensor, ndarray, List[Number], Number],
                 input_b: Union[Tensor, ndarray, List[Number], Number]):

        super(_MatrixMul, self).__init__(input_a, input_b)

        # Assert valid dimensions for matrix multiplication
        assert self.children[0].value.shape[-1] ==\
               self.children[1].value.shape[0]

    def forward(self) -> ndarray:
        return self.children[0].value @ self.children[1].value

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        if idx == 0:
            return accum_grad @ self.children[1].value.T
        else:
            return (accum_grad.T @ self.children[0].value).T


# ====================================================================================================

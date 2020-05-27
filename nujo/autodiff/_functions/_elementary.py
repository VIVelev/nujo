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
    def __init__(self,
                 input_a: Union[Tensor, ndarray, List[Number], Number],
                 input_b: Union[Tensor, ndarray, List[Number], Number],
                 name='Add'):

        super(_Addition, self).__init__(input_a,
                                        input_b,
                                        name=self.__class__.__name__)

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

    def backward(self, idx: int, acumm_grad: Function.T) -> Function.T:
        return acumm_grad * ones(self.children[idx].shape)


# ====================================================================================================


class _Negation(Function):
    def __init__(self,
                 input: Union[Tensor, ndarray, List[Number], Number],
                 name='Neg'):

        super(_Negation, self).__init__(input, name=self.__class__.__name__)

    def forward(self) -> ndarray:
        return -self.children[0].value

    def backward(self, idx: int, acumm_grad: Function.T) -> Function.T:
        return acumm_grad * -ones(self.children[0].shape)


# ====================================================================================================


class _Multiplication(Function):
    def __init__(self,
                 input_a: Union[Tensor, ndarray, List[Number], Number],
                 input_b: Union[Tensor, ndarray, List[Number], Number],
                 name='Mul'):

        super(_Multiplication, self).__init__(input_a,
                                              input_b,
                                              name=self.__class__.__name__)

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

    def backward(self, idx: int, acumm_grad: Function.T) -> Function.T:
        if idx == 0:
            return acumm_grad * self.children[1].value
        else:
            return acumm_grad * self.children[0].value


# ====================================================================================================


class _Reciprocal(Function):
    def __init__(self,
                 input: Union[Tensor, ndarray, List[Number], Number],
                 name='Recipr',
                 eps=1e-18):

        super(_Reciprocal, self).__init__(input, name=self.__class__.__name__)
        self.eps = eps

    def forward(self) -> ndarray:
        return 1 / (self.children[0].value + self.eps)

    def backward(self, idx: int, acumm_grad: Function.T) -> Function.T:
        return acumm_grad * -1 / ((self.children[0].value + self.eps)**2)


# ====================================================================================================


class _Power(Function):
    def __init__(self,
                 input_a: Union[Tensor, ndarray, List[Number], Number],
                 input_b: Union[Tensor, ndarray, List[Number], Number],
                 name='Pow'):

        super(_Power, self).__init__(input_a,
                                     input_b,
                                     name=self.__class__.__name__)

    def forward(self) -> ndarray:
        return self.children[0].value**self.children[1].value

    def backward(self, idx: int, acumm_grad: Function.T) -> Function.T:
        # TODO: FIX wrong partial - the second

        if idx == 0:
            return acumm_grad * self.children[1].value *\
                    self.children[0].value**(self.children[1].value - 1)
        else:
            return Tensor(1)


# ====================================================================================================


class _Logarithm(Function):
    def __init__(self,
                 input_a: Union[Tensor, ndarray, List[Number], Number],
                 input_b: Union[Tensor, ndarray, List[Number], Number],
                 name='Log'):

        super(_Logarithm, self).__init__(input_a,
                                         input_b,
                                         name=self.__class__.__name__)

        assert (self.children[0].value > 0).all()  # argument value limit
        assert (self.children[1].value > 0).all()  # base value limit
        assert (self.children[1].value != 0).all()  # base value limit

    def forward(self) -> ndarray:
        return log(self.children[0].value) / log(self.children[1].value)

    def backward(self, idx: int, acumm_grad: Function.T) -> Function.T:
        if idx == 0:
            return acumm_grad /\
                    (self.children[0].value * log(self.children[1].value))
        else:
            return Tensor(1)


# ====================================================================================================


class _MatrixMul(Function):
    def __init__(self,
                 input_a: Union[Tensor, ndarray, List[Number], Number],
                 input_b: Union[Tensor, ndarray, List[Number], Number],
                 name='MatMul'):

        super(_MatrixMul, self).__init__(input_a,
                                         input_b,
                                         name=self.__class__.__name__)

        # Assert valid dimensions for matrix multiplication
        assert self.children[0].value.shape[-1] ==\
               self.children[1].value.shape[0]

    def forward(self) -> ndarray:
        return self.children[0].value @ self.children[1].value

    def backward(self, idx: int, acumm_grad: Function.T) -> Function.T:
        if idx == 0:
            return acumm_grad @ self.children[1].value.T
        else:
            return (acumm_grad.T @ self.children[0].value).T


# ====================================================================================================

from math import e
from numbers import Number
from typing import List, Union

from numpy import exp, max, maximum, ndarray, ones, sum, zeros

from nujo.autodiff.function import Function
from nujo.autodiff.tensor import Tensor

__all__ = [
    '_BinaryStep',
    '_Sigmoid',
    '_TanH',
    '_ReLU',
    '_LeakyReLU',
    '_Swish',
    '_Softmax',
]

# ====================================================================================================
# Built-in Neural Network Activation Functions
#  - efficient implementation of various neural activation functions
# ====================================================================================================


class _BinaryStep(Function):
    def __init__(self,
                 input: Union[Tensor, ndarray, List[Number], Number],
                 threshold=0.5):

        super(_BinaryStep, self).__init__(input)
        self.threshold = threshold

    def forward(self) -> ndarray:
        output = zeros(self.children[0].shape)
        output[self.children[0].value > self.threshold] = 1
        return output

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad * zeros(self.children[0].shape)


# ====================================================================================================


class _Sigmoid(Function):
    def __init__(self, input: Union[Tensor, ndarray, List[Number], Number]):
        super(_Sigmoid, self).__init__(input)
        self._output: ndarray = None  # Used to compute the derivative

    def forward(self) -> ndarray:
        self._output = 1 / (1 + exp(-self.children[0].value))
        return self._output

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad * self._output * (1 - self._output)


# ====================================================================================================


class _TanH(Function):
    def __init__(self, input: Union[Tensor, ndarray, List[Number], Number]):
        super(_TanH, self).__init__(input)
        self._output: ndarray = None  # Used to compute the derivative

    def forward(self) -> ndarray:
        ''' (2 / (1 + e ^ -2x)) - 1 is equivalent to
        (e ^ x - e ^ -x) / (e ^ x + e ^ -x) it is just a
        more optimal way to compute the TanH function.

        '''

        self._output = (2 / (1 + exp(-2 * self.children[0].value))) - 1
        return self._output

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad * (1 - self._output**2)


# ====================================================================================================


class _ReLU(Function):
    def __init__(self, input: Union[Tensor, ndarray, List[Number], Number]):
        super(_ReLU, self).__init__(input)

    def forward(self) -> ndarray:
        return self.children[0].value * (self.children[0].value > 0)

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad * ones(
            self.children[0].shape) * (self.children[0].value > 0)


# ====================================================================================================


class _LeakyReLU(Function):
    def __init__(self,
                 input: Union[Tensor, ndarray, List[Number], Number],
                 eps=0.1):

        super(_LeakyReLU, self).__init__(input)
        self.eps = eps

    def forward(self) -> ndarray:
        return maximum(self.eps * self.children[0].value,
                       self.children[0].value)

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        dinput = ones(self.children[0].shape)
        dinput[self.children[0].value < 0] = self.eps
        return accum_grad * dinput


# ====================================================================================================


class _Swish(Function):
    ''' More info here: https://arxiv.org/abs/1710.05941
    '''
    def __init__(self,
                 input: Union[Tensor, ndarray, List[Number], Number],
                 beta=1):

        super(_Swish, self).__init__(input)
        self.beta = beta

        # Reuse the sigmoid activation function
        self._sigmoid = _Sigmoid(beta * input.value)
        self._output: ndarray = None  # Used to compute the derivative

    def forward(self) -> ndarray:
        self._output = self.children[0].value * self._sigmoid.forward()
        return self._output

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad * (self._output + self._sigmoid._output *
                             (1 - self._output))


# ====================================================================================================


class _Softmax(Function):
    ''' More info here:
    https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/

    '''
    def __init__(self,
                 input: Union[Tensor, ndarray, List[Number], Number],
                 dim=0,
                 base: float = e):

        super(_Softmax, self).__init__(input)

        self.dim = dim
        self.base = base

        self._output: ndarray = None  # Used to compute the derivative

    def forward(self) -> ndarray:
        # The max element of the input vector will be
        # substracted from the inputs for numerical stability.
        # This will not change the relative output of the softmax.

        exps = self.base**(
            self.children[0].value -
            max(self.children[0].value, axis=self.dim, keepdims=True))

        sums = sum(exps, axis=self.dim, keepdims=True)

        self._output = exps / sums
        return self._output

    def backward(self, idx: int, accum_grad: Function.T) -> Function.T:
        return accum_grad * self._output * (1 - self._output)


# ====================================================================================================

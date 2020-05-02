from numpy import (diag, exp, eye, hstack, maximum, ndarray, ones, repeat, sum,
                   zeros)

from nujo._typing import Union, _numerical
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
                 input: Union[Tensor, _numerical],
                 threshold=0.5,
                 name='BinaryStep'):
        super(_BinaryStep, self).__init__(input, name=name)
        self.threshold = threshold

    def forward(self) -> ndarray:
        output = zeros(self.children[0].shape)
        output[self.children[0].value > self.threshold] = 1
        return output

    def backward(self) -> tuple:
        return zeros(self.children[0].shape),


# ====================================================================================================


class _Sigmoid(Function):
    def __init__(self, input: Union[Tensor, _numerical], name='Sigmoid'):
        super(_Sigmoid, self).__init__(input, name=name)
        self._output: ndarray = None  # Used to compute the derivative

    def forward(self) -> ndarray:
        self._output = 1 / (1 + exp(-self.children[0].value))
        return self._output

    def backward(self) -> tuple:
        return self._output * (1 - self._output),


# ====================================================================================================


class _TanH(Function):
    def __init__(self, input: Union[Tensor, _numerical], name='TanH'):
        super(_TanH, self).__init__(input, name=name)
        self._output: ndarray = None  # Used to compute the derivative

    def forward(self) -> ndarray:
        ''' (2 / (1 + e ^ -2x)) - 1 is equivalent to
        (e ^ x - e ^ -x) / (e ^ x + e ^ -x) it is just a
        more optimal way to compute the TanH function.
        '''

        self._output = (2 / (1 + exp(-2 * self.children[0].value))) - 1
        return self._output

    def backward(self) -> tuple:
        return 1 - self._output**2,


# ====================================================================================================


class _ReLU(Function):
    def __init__(self, input: Union[Tensor, _numerical], name='ReLU'):
        super(_ReLU, self).__init__(input, name=name)

    def forward(self) -> ndarray:
        return self.children[0].value * (self.children[0].value > 0)

    def backward(self) -> tuple:
        return ones(self.children[0].shape) * (self.children[0].value > 0),


# ====================================================================================================


class _LeakyReLU(Function):
    def __init__(self,
                 input: Union[Tensor, _numerical],
                 eps=0.1,
                 name='LeakyReLU'):
        super(_LeakyReLU, self).__init__(input, name=name)
        self.eps = eps

    def forward(self) -> ndarray:
        return maximum(self.eps * self.children[0].value,
                       self.children[0].value)

    def backward(self) -> tuple:
        dinput = ones(self.children[0].shape)
        dinput[self.children[0].value < 0] = self.eps
        return dinput,


# ====================================================================================================


class _Swish(Function):
    ''' More info here: https://arxiv.org/abs/1710.05941
    '''
    def __init__(self, input: Union[Tensor, _numerical], beta=1, name='Swish'):
        super(_Swish, self).__init__(input, name=name)
        self.beta = beta

        self._sigmoid = _Sigmoid(
            beta * input.value)  # Reuse the sigmoid activation function
        self._output: ndarray = None  # Used to compute the derivative

    def forward(self) -> ndarray:
        self._output = self.children[0].value * self._sigmoid.forward()
        return self._output

    def backward(self) -> tuple:
        return self._output + self._sigmoid._output * (1 - self._output),


# ====================================================================================================


class _Softmax(Function):
    def __init__(self, input: Union[Tensor, _numerical], name='Softmax'):
        super(_Softmax, self).__init__(input, name=name)
        self._output: ndarray = None  # Used to compute the derivative

    def forward(self) -> ndarray:
        exps = exp(self.children[0].value)
        sums = sum(exps, axis=0, keepdims=True)

        self._output = exps / sums
        return self._output

    def backward(self) -> tuple:
        ''' Computes the Jacobian matrix

        See here:
        https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/
        for more info on how this Jacobian was computed.
        '''

        # TODO: Is there a more optimal way to compute Si matrix?
        # Should you really compute the whole Jacobian?
        # Test against PyTorch

        k, n = self._output.shape

        # Repeat each activation vector (each sample) k times
        Sj_matrix = repeat(self._output, k, axis=1)

        # Transpose each k by k matrix individually (for each sample)
        Si_matrix = hstack(
            [Sj_matrix[:, (i - k):i].T for i in range(k, (k * n) + 1, k)])

        # Make a global diagonal matrix
        # (where the diag matrix for each sample is contained)
        Sj_diag = hstack([diag(self._output[:, i]) for i in range(n)])

        # Compute the Jacobian
        jacobian = Sj_diag - Si_matrix * Sj_matrix

        # Get the needed columns, ignore the rest
        identity = eye(k * n)
        mask = hstack(
            [identity[:, i].reshape(-1, 1) for i in range(0, k * n, k)])

        return jacobian @ mask,


# ====================================================================================================

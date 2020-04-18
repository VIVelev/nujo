from math import e

from numpy import log, maximum, ndarray, ones, zeros

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
    '_BinaryStep',
    '_Sigmoid',
    '_TanH',
    '_ReLU',
    '_LeakyReLU',
    '_Swish',
]

# ====================================================================================================


class _Addition(Function):
    def __init__(self,
                 input_a: Tensor or ndarray,
                 input_b: Tensor or ndarray,
                 name='Add'):
        super(_Addition, self).__init__(input_a, input_b, name=name)

    def forward(self) -> ndarray:
        return self.children[0].value + self.children[1].value

    def backward(self) -> tuple:
        return 1, 1


# ====================================================================================================


class _Negation(Function):
    def __init__(self, input: Tensor or ndarray, name='Neg'):
        super(_Negation, self).__init__(input, name=name)

    def forward(self) -> ndarray:
        return -self.children[0].value

    def backward(self) -> tuple:
        return -1,


# ====================================================================================================


class _Multiplication(Function):
    def __init__(self,
                 input_a: Tensor or ndarray,
                 input_b: Tensor or ndarray,
                 name='Mul'):
        super(_Multiplication, self).__init__(input_a, input_b, name=name)

    def forward(self) -> ndarray:
        return self.children[0].value * self.children[1].value

    def backward(self) -> tuple:
        return self.children[1].value, self.children[0].value


# ====================================================================================================


class _Reciprocal(Function):
    def __init__(self, input: Tensor or ndarray, name='Recipr', eps=1e-18):
        super(_Reciprocal, self).__init__(input, name=name)
        self.eps = eps

    def forward(self) -> ndarray:
        return 1 / (self.children[0].value + self.eps)

    def backward(self) -> tuple:
        return -1 / ((self.children[0].value + self.eps)**2),


# ====================================================================================================


class _Power(Function):
    def __init__(self,
                 input_a: Tensor or ndarray,
                 input_b: Tensor or ndarray,
                 name='Pow'):
        super(_Power, self).__init__(input_a, input_b, name=name)

    def forward(self) -> ndarray:
        return self.children[0].value**self.children[1].value

    def backward(self) -> tuple:
        # TODO: FIX wrong partial - the second

        return (self.children[1].value *
                self.children[0].value**(self.children[1].value - 1), 1)


# ====================================================================================================


class _Logarithm(Function):
    def __init__(self,
                 input_a: Tensor or ndarray,
                 input_b: Tensor or ndarray,
                 name='Log'):
        super(_Logarithm, self).__init__(input_a, input_b, name=name)

        assert (self.children[0] > 0).all()  # argument value limit
        assert (self.children[1] > 0).all()  # base value limit
        assert (self.children[1] != 0).all()  # base value limit

    def forward(self) -> ndarray:
        return log(self.children[0].value) / log(self.children[1].value)

    def backward(self) -> tuple:
        return 1 / (self.children[0].value * log(self.children[1].value)), 1


# ====================================================================================================


class _MatrixMul(Function):
    def __init__(self,
                 input_a: Tensor or ndarray,
                 input_b: Tensor or ndarray,
                 name='MatMul'):
        super(_MatrixMul, self).__init__(input_a, input_b, name=name)

    @staticmethod
    def differentiate(X: ndarray, W: ndarray) -> tuple:
        ''' Calculate Matrix partial derivatives

        Given Z = XW, Calculate:
            - dX = dZ/dX
            - dW = dZ/dW

        Parameters:
        -----------
        X : matrix, left hand multiplier
        W : matrix, right hand multiplier

        Returns:
        --------
        dX : partial derivative of Z w.r.t. X
        dW : partial derivative of Z w.r.t. W

        '''

        # ------------------------------------------------------------
        dX = ones((X.shape[0]**2, W.shape[1] * X.shape[1]))

        i, j = 0, 0  # indecies of Z
        k, m = 0, 0  # indecies of X
        # p, q : indecies of dX

        for p in range(dX.shape[0]):
            for q in range(dX.shape[1]):
                if k == i:
                    dX[p, q] = W[m, j]

                j = q % W.shape[1]
                m = q % X.shape[1]

            i = q % X.shape[0]
            k = p % X.shape[0]

        # ------------------------------------------------------------
        dW = ones((X.shape[0] * W.shape[0], W.shape[1]**2))

        i, j = 0, 0  # indecies of Z
        k, m = 0, 0  # indecies of W
        # p, q : indecies of dW

        for p in range(dW.shape[0]):
            for q in range(dW.shape[1]):
                if m == j:
                    dW[p, q] = X[i, k]

                j = q % W.shape[1]
                m = q % W.shape[1]

            i = q % X.shape[0]
            k = p % W.shape[0]

        ##############################################################

        return dX, dW

    def forward(self) -> ndarray:
        assert self.children[0].shape[1] == self.children[1].shape[0]
        return self.children[0].value @ self.children[1].value

    def backward(self) -> tuple:
        return _MatrixMul.differentiate(*self.children)


# ====================================================================================================
# Built-in Neural Network Activation Functions
#  - efficient implementation of various neural activation functions
# ====================================================================================================


class _BinaryStep(Function):
    def __init__(self,
                 input: Tensor or ndarray,
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
    def __init__(self, input: Tensor or ndarray, name='Sigmoid'):
        super(_Sigmoid, self).__init__(input, name=name)
        self._output: ndarray = None  # Used to compute the derivative

    def forward(self) -> ndarray:
        self._output = 1 / (1 + e**-self.children[0].value)
        return self._output

    def backward(self) -> tuple:
        if (self._output == 0).all():
            print('WARNING: The forward pass of Sigmoid resulted in a zero! \
                The gradient will be zero!')
        elif (self._output == 1).all():
            print('WARNING: The forward pass of Sigmoid resulted in a one! \
                The gradient will be zero!')

        return self._output * (1 - self._output),


# ====================================================================================================


class _TanH(Function):
    def __init__(self, input: Tensor or ndarray, name='TanH'):
        super(_TanH, self).__init__(input, name=name)
        self._output: ndarray = None  # Used to compute the derivative

    def forward(self) -> ndarray:
        ''' (2 / (1 + e ^ -2x)) - 1 is equivalent to (e ^ x - e ^ -x) / (e ^ x + e ^ -x)
        it is just a more optimal way to compute the TanH function.
        '''

        self._output = (2 / (1 + e**(-2 * self.children[0].value))) - 1
        return self._output

    def backward(self) -> tuple:
        if (self._output == 1).all():
            print('WARNING: The forward pass of TanH resulted in a one! \
                The gradient will be zero!')

        return 1 - self._output**2,


# ====================================================================================================


class _ReLU(Function):
    def __init__(self, input: Tensor or ndarray, name='ReLU'):
        super(_ReLU, self).__init__(input, name=name)

    def forward(self) -> ndarray:
        return self.children[0].value * (self.children[0].value > 0)

    def backward(self) -> tuple:
        return ones(self.children[0].shape) * (self.children[0].value > 0),


# ====================================================================================================


class _LeakyReLU(Function):
    def __init__(self, input: Tensor or ndarray, eps=0.1, name='LeakyReLU'):
        super(_LeakyReLU, self).__init__(input, name=name)
        self.eps = eps

    def forward(self) -> ndarray:
        # TODO: Can this be done in a more efficient way?
        return maximum(self.eps * self.children[0].value,
                       self.children[0].value)

    def backward(self) -> tuple:
        dinput = ones(self.children[0].shape)
        dinput[self.children[0].value < 0] = self.eps
        return dinput,


# ====================================================================================================


class _Swish(Function):
    def __init__(self, input: Tensor or ndarray, beta=1, name='Swish'):
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

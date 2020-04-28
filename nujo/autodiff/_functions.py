from numpy import (diag, exp, eye, hstack, log, maximum, ndarray, ones, prod,
                   repeat, sum, zeros)

from nujo._typing import Union, _numerical
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
    '_InnerSum',
    '_InnerProd',
    '_BinaryStep',
    '_Sigmoid',
    '_TanH',
    '_ReLU',
    '_LeakyReLU',
    '_Swish',
    '_Softmax',
]

# ====================================================================================================


class _Addition(Function):
    def __init__(self,
                 input_a: Union[Tensor, _numerical],
                 input_b: Union[Tensor, _numerical],
                 name='Add'):
        super(_Addition, self).__init__(input_a, input_b, name=name)

    def forward(self) -> ndarray:
        return self.children[0].value + self.children[1].value

    def backward(self) -> tuple:
        return 1, 1


# ====================================================================================================


class _Negation(Function):
    def __init__(self, input: Union[Tensor, _numerical], name='Neg'):
        super(_Negation, self).__init__(input, name=name)

    def forward(self) -> ndarray:
        return -self.children[0].value

    def backward(self) -> tuple:
        return -1,


# ====================================================================================================


class _Multiplication(Function):
    def __init__(self,
                 input_a: Union[Tensor, _numerical],
                 input_b: Union[Tensor, _numerical],
                 name='Mul'):
        super(_Multiplication, self).__init__(input_a, input_b, name=name)

    def forward(self) -> ndarray:
        return self.children[0].value * self.children[1].value

    def backward(self) -> tuple:
        return self.children[1].value, self.children[0].value


# ====================================================================================================


class _Reciprocal(Function):
    def __init__(self,
                 input: Union[Tensor, _numerical],
                 name='Recipr',
                 eps=1e-18):
        super(_Reciprocal, self).__init__(input, name=name)
        self.eps = eps

    def forward(self) -> ndarray:
        return 1 / (self.children[0].value + self.eps)

    def backward(self) -> tuple:
        return -1 / ((self.children[0].value + self.eps)**2),


# ====================================================================================================


class _Power(Function):
    def __init__(self,
                 input_a: Union[Tensor, _numerical],
                 input_b: Union[Tensor, _numerical],
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
                 input_a: Union[Tensor, _numerical],
                 input_b: Union[Tensor, _numerical],
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
                 input_a: Union[Tensor, _numerical],
                 input_b: Union[Tensor, _numerical],
                 name='MatMul'):
        super(_MatrixMul, self).__init__(input_a, input_b, name=name)

    @staticmethod
    def differentiate(X: 'Tensor', W: 'Tensor') -> tuple:
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
        assert self.children[0].shape[-1] == self.children[1].shape[0]
        return self.children[0].value @ self.children[1].value

    def backward(self) -> tuple:
        return _MatrixMul.differentiate(*self.children)


# ====================================================================================================
# Inner aggregate functions
# ====================================================================================================


class _InnerSum(Function):
    def __init__(self,
                 input: Union[Tensor, _numerical],
                 dim: int = None,
                 keepdim=False,
                 name='InnerSum'):
        super(_InnerSum, self).__init__(input, name=name)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self) -> ndarray:
        return sum(self.children[0].value,
                   axis=self.dim,
                   keepdims=self.keepdim)

    def backward(self) -> tuple:
        return ones(self.children[0].shape),


class _InnerProd(Function):
    def __init__(self,
                 input: Union[Tensor, _numerical],
                 dim: int = None,
                 keepdim=False,
                 name='InnerProd'):
        super(_InnerProd, self).__init__(input, name=name)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self) -> ndarray:
        return prod(self.children[0].value,
                    axis=self.dim,
                    keepdims=self.keepdim)

    def backward(self) -> tuple:
        mask = -(eye(self.children[0].shape[0]) - 1)
        matrix = self.children[0].value.repeat(self.children[0].shape[0],
                                               axis=-1)

        return prod(mask * matrix, axis=0),


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
        # TODO: Can this be done in a more efficient way?
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
        return Sj_diag - Si_matrix * Sj_matrix,


# ====================================================================================================

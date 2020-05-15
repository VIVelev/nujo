''' Neural Network activation functions

More info here:
https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/

'''

from nujo.autodiff._functions._activations import (_BinaryStep, _LeakyReLU,
                                                   _ReLU, _Sigmoid, _Softmax,
                                                   _Swish, _TanH)
from nujo.autodiff.tensor import Tensor
from nujo.flow import Flow

__all__ = [
    'BinaryStep',
    'Sigmoid',
    'TanH',
    'ReLU',
    'LeakyReLU',
    'Swish',
    'Softmax',
]

# ====================================================================================================
# Scalar (Single-class) activation functions
# ====================================================================================================


class BinaryStep(Flow):
    ''' Binary step function

        if val > threshold:
            return 1
        else:
            return 0

    '''
    def __init__(self, threshold=0.5, name='BinaryStep'):
        super(BinaryStep, self).__init__(name=name)
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        return _BinaryStep(x, threshold=self.threshold)()


# ====================================================================================================


class Sigmoid(Flow):
    ''' Sigmoid activation function

        sigmoid(x) = 1 / (1 + e ^ -x)

    '''
    def __init__(self, name='Sigmoid'):
        super(Sigmoid, self).__init__(name=name)

    def forward(self, x: Tensor) -> Tensor:
        return _Sigmoid(x)()


# ====================================================================================================


class TanH(Flow):
    ''' TanH activation function

        tanh(x) = (e ^ x - e ^ -x) /  (e ^ x + e ^ -x)

    '''
    def __init__(self, name='TanH'):
        super(TanH, self).__init__(name=name)

    def forward(self, x: Tensor) -> Tensor:
        return _TanH(x)()


# ====================================================================================================


class ReLU(Flow):
    ''' ReLU (Rectified Linear Unit) activation function

        relu(x) = max(0, x)

    '''
    def __init__(self, name='ReLU'):
        super(ReLU, self).__init__(name=name)

    def forward(self, x: Tensor) -> Tensor:
        return _ReLU(x)()


# ====================================================================================================


class LeakyReLU(Flow):
    ''' Leaky ReLU activation function

        leaky_relu = max(eps * x, x)

    '''
    def __init__(self, eps=0.1, name='LeakyReLU'):
        super(LeakyReLU, self).__init__(name=name)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return _LeakyReLU(x, eps=self.eps)()


# ====================================================================================================


class Swish(Flow):
    ''' Swish activation function

        swish(x) = x * sigmoid(beta * x) = x / (1 + e ^ (-beta * x))

    "Searching for Activation Functions"
    Prajit Ramachandran, Barret Zoph, Quoc V. Le
    (https://arxiv.org/abs/1710.05941)

    '''
    def __init__(self, beta=1, name='Swish'):
        super(Swish, self).__init__(name=name)
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        return _Swish(x, beta=self.beta)()


# ====================================================================================================
# Vector (Multi-class) activation functions
# ====================================================================================================


class Softmax(Flow):
    ''' Softmax activation function

        softmax(z) = e ^ z_i / sum(e ^ z_i)

    Nice read here:
    https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/

    '''
    def __init__(self, name='Softmax'):
        super(Softmax, self).__init__(name=name)

    def forward(self, x: Tensor) -> Tensor:
        return _Softmax(x)()


# ====================================================================================================

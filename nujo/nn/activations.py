''' Neural Network activation functions

More info here: https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/
'''

from nujo.autodiff._functions import _Sigmoid, _TanH
from nujo.flow import Flow

__all__ = [
    'BinaryStep',
    'Sigmoid',
    'TanH',
    'ReLU',
    'LeakyReLU',
    'Softmax',
    'Swish',
]

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

    def forward(self, x):
        if x > self.threshold:
            return 1
        else:
            return 0


# ====================================================================================================


class Sigmoid(Flow):
    ''' Sigmoid activation function

        sigmoid(x) = 1 / (1 + e ^ -x)

    '''
    def __init__(self, name='Sigmoid'):
        super(Sigmoid, self).__init__(name=name)

    def forward(self, x):
        return _Sigmoid(x)()


# ====================================================================================================


class TanH(Flow):
    ''' TanH activation function

        tanh(x) = (e ^ x - e ^ -x) /  (e ^ x + e ^ -x)

    '''
    def __init__(self, name='TanH'):
        super(TanH, self).__init__(name=name)

    def forward(self, x):
        return _TanH(x)()


# ====================================================================================================

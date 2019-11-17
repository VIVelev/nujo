import numpy as np

from .base import Transformation
from ..core.autodiff import Variable

__all__ = [
    'Linear',
]


class Linear(Transformation):
    '''Linear Transformation

        f(x) = xW + b
    '''

    def __init__(self, in_features, out_features, bias=True, name='Linear'):
        super(Linear, self).__init__(name=f'{name}({in_features}, {out_features})')

        self.weights = Variable(
            np.random.randn(in_features, out_features),
            name=self.name+'.weights',
        )

        if bias:
            self.bias = Variable(
                np.random.randn(1, out_features),
                name=self.name+'.bias',
            )
        else:
            self.bias = 0

    def forward(self, input):
        return input@self.weights + self.bias

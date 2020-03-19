from numpy.random import randn

from nujo.autodiff.tensor import Tensor
from nujo.nn.base import Transformation

__all__ = [
    'Linear',
]


class Linear(Transformation):
    '''Linear Transformation

        f(x) = xW + b
    '''
    def __init__(self, in_features, out_features, bias=True, name='Linear'):
        super(Linear,
              self).__init__(name=f'{name}({in_features}, {out_features})')

        weights = Tensor(randn(in_features, out_features),
                         name=self.name + '.weights')

        if bias:
            bias = Tensor(randn(1, out_features), name=self.name + '.bias')
        else:
            bias = Tensor(0)

        self.parameters = [weights, bias]

    def forward(self, input):
        return input @ self.parameters[0] + self.parameters[1]

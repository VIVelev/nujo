from numpy import ndarray

from nujo.autodiff.tensor import Tensor
from nujo.flow import Flow
from nujo.init import randn, zeros

__all__ = [
    'Linear',
]


class Linear(Flow):
    '''Linear Layer

        f(x) = xW + b
    '''
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias=True,
                 name='Linear'):
        super(Linear,
              self).__init__(name=f'{name}({in_features}, {out_features})')

        self.W = randn(in_features, out_features, name=self.name + '.W')

        if bias:
            self.b = randn(1, out_features, name=self.name + '.bias')
        else:
            self.b = zeros(diff=False)

    def forward(self, x: Tensor or ndarray) -> Tensor:
        x = x if isinstance(x, Tensor) else Tensor(x, name='x', diff=False)
        return x @ self.W + self.b

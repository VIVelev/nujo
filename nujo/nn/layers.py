from nujo.autodiff.tensor import Tensor
from nujo.flow import Flow
from nujo.init import randn

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

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.W = randn(self.in_features,
                       self.out_features,
                       name=self.name + '.W')

        if self.bias:
            self.b = randn(1, self.out_features, name=self.name + '.bias')

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.W
        return out + self.b if self.bias else out

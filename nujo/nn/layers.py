from nujo.autodiff.tensor import Tensor
from nujo.flow import Flow
from nujo.init import randn

__all__ = [
    'Linear',
]


class Linear(Flow):
    ''' Linear Layer

        f(x) = Wx + b

    Parameters:
    -----------
     - in_features : int, dim of input variables
     - out_features : int, wanted dim of output variables
     - bias : bool, whether to train a bias term or no
     - name : string, identifier for the current layer

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

        self.W = randn(self.out_features,
                       self.in_features,
                       name=self.name + '.W')

        if self.bias:
            self.b = randn(self.out_features, 1, name=self.name + '.bias')

    def forward(self, x: Tensor) -> Tensor:
        out = self.W @ x
        return out + self.b if self.bias else out

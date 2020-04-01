from nujo.flow import Flow
from nujo.init import randn, zeros

__all__ = [
    'Linear',
]


class Linear(Flow):
    '''Linear Layer

        f(x) = xW + b
    '''
    def __init__(self, in_features, out_features, bias=True, name='Linear'):
        super(Linear,
              self).__init__(name=f'{name}({in_features}, {out_features})')

        self.W = randn(in_features, out_features, name=self.name + '.W')

        if bias:
            self.b = randn(1, out_features, name=self.name + '.bias')
        else:
            self.b = zeros(diff=False)

    def forward(self, x):
        return x @ self.W + self.b

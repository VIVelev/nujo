import numpy as np

from .base import Optimizer

__all__ = [
    'GradientDescent',
    # 'Momentum',
    # 'RMSprop',
    # 'Adam',
]


class GradientDescent(Optimizer):
    def __init__(self, net, lr=0.1):
        super(GradientDescent, self).__init__(net, lr)

    def step(self):
        pass

import numpy as np

from nujo.autodiff import no_diff
from nujo.optim.base import Optimizer

__all__ = [
    'GradientDescent',
    'Momentum',
    # 'RMSprop',
    # 'Adam',
]


# ====================================================================================================
# ====================================================================================================

class GradientDescent(Optimizer):
    def __init__(self, parameters, lr=0.1):
        super(GradientDescent, self).__init__(parameters, lr)

    def step(self):
        with no_diff():
            for l in range(len(self.parameters)):
                for i in range(len(self.parameters[l])):
                    self.parameters[l][i] -= self.lr * self.parameters[l][i].grad

# ====================================================================================================

class Momentum(Optimizer):
    def __init__(self, parameters, lr=0.1, beta=0.9):
        pass

    def step(self):
        pass

# ====================================================================================================

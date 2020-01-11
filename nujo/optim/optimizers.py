''' Neural Network Optimizers

    Check out the following link for more info about the optimizers: 
    http://ruder.io/optimizing-gradient-descent/index.html
'''

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
            for l in range(len(self.parameters)):   # Iterate over layers
                for i in range(len(self.parameters[l])):    # Iterate over parameters in layer `l`
                    self.parameters[l][i] -= self.lr * self.parameters[l][i].grad

# ====================================================================================================

class Momentum(Optimizer):

    def __init__(self, parameters, lr=0.1, beta=0.9):
        super(Momentum, self).__init__(parameters, lr)

        self.beta = beta
        self.velocity = {}

    def step(self):
        with no_diff():
            for l in range(len(self.parameters)):   # Iterate over layers
                for i in range(len(self.parameters[l])):    # Iterate over parameters in layer `l`

                    # Get the corresponding velocity
                    key = f'Layer[{l}]-Param[{i}]'
                    if key not in self.velocity:
                        self.velocity[key] = np.zeros_like(self.parameters[l][i])

                    # Exponentially Weighted Moving Average
                    self.velocity[key] = self.beta * self.velocity[key] \
                                            + (1 - self.beta) * self.parameters[l][i].grad
                    # Update
                    self.parameters[l][i] -= self.lr * self.velocity[key]

# ====================================================================================================

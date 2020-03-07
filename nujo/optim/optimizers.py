''' Stochastic Gradient Descent (SGD) Optimizers

    Check out the following link for more info about the optimizers:
    http://ruder.io/optimizing-gradient-descent/index.html
'''

from numpy import sqrt, square, zeros_like

from nujo.autodiff import no_diff
from nujo.optim.base import Optimizer

__all__ = [
    'GradientDescent',
    'Momentum',
    'RMSprop',
    'Adam',
]

# ====================================================================================================


class GradientDescent(Optimizer):
    def __init__(self, params, lr=0.001):
        super(GradientDescent, self).__init__(params, lr)

    def step(self):
        with no_diff():
            for l in range(len(self.params)):  # Iterate over layers

                # Iterate over params in layer `l`
                for i in range(len(self.params[l])):
                    self.params[l][i] -= self.lr * self.params[l][i].grad


# ====================================================================================================


class Momentum(Optimizer):
    def __init__(self, params, lr=0.001, beta=0.9):
        super(Momentum, self).__init__(params, lr)

        self.beta = beta
        self._velocity = {}

    def step(self):
        with no_diff():
            for l in range(len(self.params)):  # Iterate over layers

                # Iterate over params in layer `l`
                for i in range(len(self.params[l])):

                    # Get the corresponding velocity
                    key = f'Layer[{l}]-Param[{i}]'
                    if key not in self._velocity:
                        self._velocity[key] = zeros_like(self.params[l][i])

                    # Exponentially Weighted Moving Average
                    self._velocity[key] = self.beta * self._velocity[key] +\
                        (1 - self.beta) * self.params[l][i].grad
                    # Update
                    self.params[l][i] -= self.lr * self._velocity[key]


# ====================================================================================================


class RMSprop(Optimizer):
    def __init__(self, params, lr=0.001, beta=0.999, eps=1e-08):
        super(RMSprop, self).__init__(params, lr)

        self.beta = beta
        self.eps = eps
        self._squared = {}

    def step(self):
        with no_diff():
            for l in range(len(self.params)):  # Iterate over layers

                # Iterate over params in layer `l`
                for i in range(len(self.params[l])):

                    # Get the corresponding squared gradient
                    key = f'Layer[{l}]-Param[{i}]'
                    if key not in self._squared:
                        self._squared[key] = zeros_like(self.params[l][i])

                    # Exponentially Weighted Moving Average
                    self._squared[key] = self.beta * self._squared[key] +\
                        (1 - self.beta) * square(self.params[l][i].grad)
                    # Update
                    self.params[l][i] -= self.lr * self.params[l][i].grad /\
                        (sqrt(self._squared[key]) + self.eps)


# ====================================================================================================


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
        super(Adam, self).__init__(params, lr)

        self.betas = betas
        self.eps = eps

        self._velocity = {}
        self._squared = {}
        self._t = 1

    def step(self):
        with no_diff():
            for l in range(len(self.params)):  # Iterate over layers

                # Iterate over params in layer `l`
                for i in range(len(self.params[l])):

                    # Get the corresponding velocity and squared gradient
                    key = f'Layer[{l}]-Param[{i}]'
                    if key not in self._velocity:
                        self._velocity[key] = zeros_like(self.params[l][i])
                        self._squared[key] = zeros_like(self.params[l][i])

                    # Exponentially Weighted Moving Average
                    self._velocity[key] = self.betas[0] * self._velocity[key] +\
                        (1 - self.betas[0]) * self.params[l][i].grad

                    self._squared[key] = self.betas[1] * self._squared[key] +\
                        (1 - self.betas[1]) * square(self.params[l][i].grad)

                    # Bias correction
                    v_corrected = self._velocity[key] /\
                        (1 - self.betas[0]**self._t)
                    s_corrected = self._squared[key] /\
                        (1 - self.betas[1]**self._t)
                    self._t += 1

                    # Update
                    self.params[l][i] -= self.lr * v_corrected /\
                        (sqrt(s_corrected) + self.eps)


# ====================================================================================================

''' Stochastic Gradient Descent (SGD) Optimizers

    Check out the following link for more info about the optimizers:
    http://ruder.io/optimizing-gradient-descent/index.html
'''

from numpy import sqrt, square, zeros_like

from nujo.optim.optimizer import Optimizer

__all__ = [
    'SGD',
    'Momentum',
    'RMSprop',
    'Adam',
]

# ====================================================================================================

# TODO: use type hints ???


class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super(SGD, self).__init__(params, lr)

    def update_rule(self, param, grad):
        return param - self.lr * grad


# ====================================================================================================


class Momentum(Optimizer):
    def __init__(self, params, lr=0.001, beta=0.9):
        super(Momentum, self).__init__(params, lr)

        self.beta = beta
        self._velocity = {}

    def update_rule(self, param, grad):
        # Get the corresponding velocity
        key = param.name
        if key not in self._velocity:
            self._velocity[key] = zeros_like(param)

        # Exponentially Weighted Moving Average
        self._velocity[key] = self.beta * self._velocity[key] +\
            (1 - self.beta) * grad

        # Update
        return param - self.lr * self._velocity[key]


# ====================================================================================================


class RMSprop(Optimizer):
    def __init__(self, params, lr=0.001, beta=0.999, eps=1e-08):
        super(RMSprop, self).__init__(params, lr)

        self.beta = beta
        self.eps = eps
        self._squared = {}

    def update_rule(self, param, grad):
        # Get the corresponding squared gradient
        key = param.name
        if key not in self._squared:
            self._squared[key] = zeros_like(param)

        # Exponentially Weighted Moving Average
        self._squared[key] = self.beta * self._squared[key] +\
            (1 - self.beta) * square(grad)

        # Update
        return param - self.lr * grad / (sqrt(self._squared[key]) + self.eps)


# ====================================================================================================


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
        super(Adam, self).__init__(params, lr)

        self.betas = betas
        self.eps = eps

        self._velocity = {}
        self._squared = {}
        self._t = 1

    def update_rule(self, param, grad):
        # Get the corresponding velocity and squared gradient
        key = param.name
        if key not in self._velocity:
            self._velocity[key] = zeros_like(param)
            self._squared[key] = zeros_like(param)

        # Exponentially Weighted Moving Average
        self._velocity[key] = self.betas[0]*self._velocity[key] +\
            (1 - self.betas[0]) * grad

        self._squared[key] = self.betas[1] * self._squared[key] +\
            (1 - self.betas[1]) * square(grad)

        # Bias correction
        v_corrected = self._velocity[key] /\
            (1 - self.betas[0]**self._t)
        s_corrected = self._squared[key] /\
            (1 - self.betas[1]**self._t)
        self._t += 1

        # Update
        return param - self.lr * v_corrected / (sqrt(s_corrected) + self.eps)


# ====================================================================================================

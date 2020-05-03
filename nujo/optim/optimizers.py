''' Stochastic Gradient Descent (SGD) Optimizers

Check out the following link for more info about the optimizers:
http://ruder.io/optimizing-gradient-descent/index.html

'''

from typing import Dict, List

from nujo.autodiff.tensor import Tensor
from nujo.init.basic import zeros_like
from nujo.math.scalar import sqrt
from nujo.optim.optimizer import Optimizer

__all__ = [
    'SGD',
    'Momentum',
    'RMSprop',
    'Adam',
]

# ====================================================================================================


class SGD(Optimizer):
    ''' SGD: Stochastic Gradient Descent

    An iterative method for optimizing an objective function.

    Parameters:
    -----------
     - params : list of Tensors, the parameters which to update
     - lr : float, the learning rate

    '''
    def __init__(self, params: List[Tensor], lr=0.005):
        super(SGD, self).__init__(params, lr)

    def update_rule(self, param: Tensor, grad: Tensor) -> Tensor:
        return param - self.lr * grad


# ====================================================================================================


class Momentum(Optimizer):
    ''' Momentum

    A method that helps accelerate SGD in the relevant direction and
    dampens oscillations. It does this by adding a fraction of the
    update vector of the past time step to the current update vector.

    Parameters:
    -----------
     - params : list of Tensors, the parameters which to update
     - lr : float, the learning rate
     - beta : float, the fraction of the update vector of the past
       time step to be added to the current update vector

    '''
    def __init__(self, params: List[Tensor], lr=0.001, beta=0.9):
        super(Momentum, self).__init__(params, lr)

        self.beta = beta
        self._velocity: Dict[str, Tensor] = {}

    def update_rule(self, param: Tensor, grad: Tensor) -> Tensor:
        # Get the corresponding velocity
        key = param.name
        if key not in self._velocity:
            self._velocity[key] = zeros_like(param)

        # Exponentially Weighted Moving Average
        self._velocity[key] = self.beta * self._velocity[key] +\
            (1 - self.beta) * grad

        # Update rule
        return param - self.lr * self._velocity[key]


# ====================================================================================================


class RMSprop(Optimizer):
    ''' RMSprop

    A gradient-based optimization technique proposed by Geoffrey Hinton
    at his Neural Networks Coursera course. It uses a moving average
    of squared gradients to normalize the gradient itself.

    Parameters:
    -----------
     - params : list of Tensors, the parameters which to update
     - lr : float, the learning rate
     - beta : float, the squared gradient coefficients
     - eps : float, added for numerical stability

    '''
    def __init__(self, params: List[Tensor], lr=0.001, beta=0.999, eps=1e-09):
        super(RMSprop, self).__init__(params, lr)

        self.beta = beta
        self.eps = eps
        self._squared: Dict[str, Tensor] = {}

    def update_rule(self, param: Tensor, grad: Tensor) -> Tensor:
        # Get the corresponding squared gradient
        key = param.name
        if key not in self._squared:
            self._squared[key] = zeros_like(param)

        # Exponentially Weighted Moving Average
        self._squared[key] = self.beta * self._squared[key] +\
            (1 - self.beta) * grad**2

        # Update rule
        return param - self.lr * grad / (sqrt(self._squared[key]) + self.eps)


# ====================================================================================================


class Adam(Optimizer):
    ''' Adam: Adaptive Moment Estimation

    Another method that computes adaptive learning rates
    for each parameter. It basically combines Momentum
    and RMSprop into one update rule.

    Parameters:
    -----------
     - params : list of Tensors, the parameters which to update
     - lr : float, the learning rate
     - betas : tuple of 2 floats, the velocity (Momentum) and
       squared gradient (RMSprop) coefficients
     - eps : float, added for numerical stability

    '''
    def __init__(self,
                 params: List[Tensor],
                 lr=0.001,
                 betas=(0.9, 0.999),
                 eps=1e-09):

        super(Adam, self).__init__(params, lr)

        self.betas = betas
        self.eps = eps

        self._velocity: Dict[str, Tensor] = {}
        self._squared: Dict[str, Tensor] = {}
        self._t = 1

    def update_rule(self, param: Tensor, grad: Tensor) -> Tensor:
        # Get the corresponding velocity and squared gradient
        key = param.name
        if key not in self._velocity:
            self._velocity[key] = zeros_like(param)
            self._squared[key] = zeros_like(param)

        # Exponentially Weighted Moving Average
        self._velocity[key] = self.betas[0]*self._velocity[key] +\
            (1 - self.betas[0]) * grad

        self._squared[key] = self.betas[1] * self._squared[key] +\
            (1 - self.betas[1]) * grad**2

        # Bias correction
        v_corrected = self._velocity[key] / (1 - self.betas[0]**self._t)
        s_corrected = self._squared[key] / (1 - self.betas[1]**self._t)
        self._t += 1

        # Update rule
        return param - self.lr * v_corrected / (sqrt(s_corrected) + self.eps)


# ====================================================================================================

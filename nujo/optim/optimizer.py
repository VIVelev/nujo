from abc import abstractmethod
from typing import Generator

from nujo.autodiff import Tensor, no_diff


class Optimizer:
    ''' Stochastic Gradient Descent Optimizer

    A base class. If you want to implement a custom
    optimizer you should inherit this class.

    The optimizers are made to work with nujo flows.

    Parameters:
    -----------
     - params : generator of Tensors, the parameters which to update
     - lr : float, the learning rate

    '''
    def __init__(self, params: Generator[Tensor, None, None], lr: float):
        self.params = params
        self.lr = lr

    @abstractmethod
    def update_rule(self, param: Tensor, grad: Tensor) -> Tensor:
        ''' Implement the update rule here. '''
        pass

    def step(self) -> None:
        ''' Updates all the parameters.
        '''

        with no_diff():
            for param in self.params():
                param <<= self.update_rule(param, param.grad)

    def zero_grad(self) -> None:
        ''' Zeros the gradients of the parameters.
        '''

        for param in self.params():
            param.zero_grad()

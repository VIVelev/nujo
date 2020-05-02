from abc import abstractmethod
from typing import Generator

from nujo.autodiff import Tensor, no_diff


class Optimizer:
    ''' Stochastic Gradient Descent Optimizer

    A base class.

    Parameters:
    -----------
    params : generator of tensors, the parameters which to update
    lr : float, the learning rate

    '''
    def __init__(self, params: Generator[Tensor, Tensor, None], lr: float):
        self.params = params
        self.lr = lr

    @abstractmethod
    def update_rule(self, param: Tensor, grad: Tensor) -> Tensor:
        pass

    def step(self) -> None:
        with no_diff():
            parameters = self.params()
            for param in parameters:
                parameters.send(self.update_rule(param, param.grad))

    def zero_grad(self) -> None:
        parameters = self.params()
        for param in parameters:
            param.zero_grad()
            parameters.send(param)

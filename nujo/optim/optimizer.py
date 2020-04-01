from abc import abstractmethod

from numpy import ndarray

from nujo.autodiff import Tensor, no_diff


class Optimizer:
    ''' Stochastic Gradient Descent Optimizer

    A base class.

    Parameters:
    -----------
    params : list of ndarray(s), the parameters which to update
    lr : float, the learning rate

    '''
    def __init__(self, params: list, lr: float) -> None:
        self.params = params
        self.lr = lr

    @abstractmethod
    def update_rule(self, param: Tensor, grad: ndarray) -> Tensor:
        pass

    def step(self) -> None:
        with no_diff():
            # Iterate over layers
            for l in range(len(self.params)):
                # Iterate over params in layer `l`
                for i in range(len(self.params[l])):

                    self.params[l][i] <<= self.update_rule(
                        self.params[l][i], self.params[l][i].grad)

    def zero_grad(self) -> None:
        # Iterate over layers
        for l in range(len(self.params)):
            # Iterate over params in layer `l`
            for i in range(len(self.params[l])):
                self.params[l][i].zero_grad()

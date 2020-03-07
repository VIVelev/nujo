from abc import abstractmethod

from nujo.autodiff import no_diff


class Optimizer:
    ''' Stochastic Gradient Descent Optimizer

    A base class.

    Parameters:
    -----------
    params : ndarray, the parameters which to update
    lr : float, the learning rate

    '''
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    @abstractmethod
    def _single_step(self, l, i):
        pass

    def step(self):
        with no_diff():
            for l in range(len(self.params)):  # Iterate over layers

                # Iterate over params in layer `l`
                for i in range(len(self.params[l])):
                    self._single_step(l, i)

    def zero_grad(self):
        for l in range(len(self.params)):  # Iterate over layers

            # Iterate over params in layer `l`
            for i in range(len(self.params[l])):
                self.params[l][i].zero_grad()

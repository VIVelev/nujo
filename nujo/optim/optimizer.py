from abc import abstractmethod


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
    def step(self):
        pass

    def zero_grad(self):
        for l in range(len(self.params)):  # Iterate over layers

            # Iterate over params in layer `l`
            for i in range(len(self.params[l])):
                self.params[l][i].zero_grad()

from abc import abstractmethod

from numpy import array

from nujo.autodiff.modes import DIFF_ENABLED


class Function:
    ''' Abstract Base Class for functions '''
    def __init__(self, *inputs, name='Function'):
        self.inputs = inputs
        self.name = name

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self, grad):
        pass

    def __call__(self):
        z = self.forward()

        if DIFF_ENABLED:
            for tensor, derivative in zip(self.inputs,
                                          self.backward(array(1))):
                tensor.dependencies.append((derivative, z))

        return z

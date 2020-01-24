from numpy import array

from nujo.autodiff.tensor import Tensor


class Constant(Tensor):
    ''' Constant Tensor

    Parameters:
    -----------
    value : value

    '''
    def __init__(self, value):
        super(Constant, self).__init__(value, name=f'{value}<Const>')

    def compute_grad(self):
        self._grad = array(1)

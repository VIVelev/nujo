from numpy import array, eye, tile

from nujo.autodiff.tensor import Tensor


class Variable(Tensor):
    ''' Variable Tensor

    Tensor for which gradient is computed.

    Parameters:
    -----------
    value : value
    name : string
    childre : list

    '''

    def __init__(self, value, name='undefined', children=[]):
        super(Variable, self).__init__(value, name, children)

    def compute_grad(self, debug=False):
        if debug:
            print()
            print('='*30)
            print(self, self.shape, '- dependencies')

        if len(self.dependencies) == 0:
            self._grad = array(1)
            return

        self._grad = 0
        for weight, z in self.dependencies:
            if debug:
                print('-'*10)
                print('Weight of `Z_prev Grad`:', weight)
                print('Shape:', weight.shape)
                print('-'*5)
                print('Z_prev Grad:', z.grad)
                print('Shape:', z.grad.shape)
                print('-'*5)
                
            if weight.shape == () or z.grad.shape == ():
                self._grad += weight * z.grad
            else:
                weight = weight.reshape(z.grad.shape[0], self.shape[0], z.grad.shape[1] * self.shape[1])
                z_grad = z.grad.repeat(self.shape[1], axis=1).reshape(z.grad.shape[0], 1, -1)
                sum_mask = tile(eye(self.shape[1]), z.grad.shape[1])
                accumulated_grad = ((weight * z_grad) @ sum_mask.T).sum(0)
                self._grad += accumulated_grad / z.grad.shape[0]
        
        if debug:
            print('Current Grad:', self._grad)
            print('Shape:', self._grad.shape)
            print('-'*5)
            print()

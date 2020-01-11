from copy import deepcopy

import numpy as np
from numpy import array

from nujo.autodiff import modes
from nujo.autodiff.utils import counter, matrix_dotprod_differentiation

__all__ = [
    'Expression',
    'Variable',
    'Constant',
]


# ====================================================================================================
# ====================================================================================================

class Expression:

    _z_counter = counter()

    def __init__(self, value, name='undefined', children=[]):
        self.value = value
        self.name = name
        self.children = children
        self.dependencies = []

        self._grad = None

    @property
    def grad(self):
        if self._grad is None:
            self.compute_grad()
        return self._grad

    @property
    def T(self):
        transposed = deepcopy(self)
        transposed.value = self.value.T
        return transposed

    @property
    def shape(self):
        self.value = array(self.value)
        return self.value.shape

    def compute_grad(self):
        pass

    def zero_grad(self):
        self.dependencies = []
        self._grad = None

    def backward(self):
        self.compute_grad()
        for child in self.children:
            child.backward()

        self._z_counter.reset()

    def __add__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            suffix = '<Add>(NO_DIFF)'
            return Variable(self.value + other.value, name=self.name+(suffix not in self.name)*suffix)

        z = Variable(self.value + other.value,
                        name=f'Z_{self._z_counter.get()}<Add>',
                        children=[self, other])

        self.dependencies.append(( array(1), z ))
        other.dependencies.append(( array(1), z ))

        return z

    def __radd__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)
        
        return other.__add__(self)

    def __sub__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            suffix = '<Sub>(NO_DIFF)'
            return Variable(self.value - other.value, name=self.name+(suffix not in self.name)*suffix)

        z = Variable(self.value - other.value,
                        name=f'Z_{self._z_counter.get()}<Sub>',
                        children=[self, other])
        
        self.dependencies.append(( array(1), z ))
        other.dependencies.append(( array(-1), z ))

        return z

    def __rsub__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)
        
        return other.__sub__(self)

    def __mul__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            suffix = '<Mul>(NO_DIFF)'
            return Variable(self.value * other.value, name=self.name+(suffix not in self.name)*suffix)

        z = Variable(self.value * other.value,
                        name=f'Z_{self._z_counter.get()}<Mul>',
                        children=[self, other])

        self.dependencies.append(( array(other.value), z ))
        other.dependencies.append(( array(self.value), z ))

        return z

    def __rmul__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)
        
        return other.__mul__(self)

    def __truediv__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            suffix = '<TrueDiv>(NO_DIFF)'
            return Variable(self.value / other.value, name=self.name+(suffix not in self.name)*suffix)

        z = Variable(self.value / other.value,
                        name=f'Z_{self._z_counter.get()}<TrueDiv>',
                        children=[self, other])

        self.dependencies.append(( array(1/other.value), z ))
        other.dependencies.append(( array((-self.value)/(other.value**2)), z ))

        return z

    def __rtruediv__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)
        
        return other.__truediv__(self)

    def __pow__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            suffix = '<Pow>(NO_DIFF)'
            return Variable(self.value ** other.value, name=self.name+(suffix not in self.name)*suffix)

        z = Variable(self.value ** other.value,
                        name=f'Z_{self._z_counter.get()}<Pow>',
                        children=[self, other])

        self.dependencies.append(( array(other.value*self.value**(other.value-1)), z ))
        other.dependencies.append(( array(1), z )) # TODO: FIX (wrong partial)

        return z

    def __matmul__(self, other):
        assert self.shape[1] == other.shape[0]

        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            suffix = '<MatMul>(NO_DIFF)'
            return Variable(self.value @ other.value, name=self.name+(suffix not in self.name)*suffix)

        z = Variable(self.value @ other.value,
                        name=f'Z_{self._z_counter.get()}<MatMul>',
                        children=[self, other])

        dself, dother = matrix_dotprod_differentiation(z, self.value, other.value)
        self.dependencies.append(( dself, z ))
        other.dependencies.append(( dother, z ))

        return z

    def __rmatmul__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        return other.__matmul__(self)

    def __repr__(self):
        return self.name

# ====================================================================================================

class Variable(Expression):

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
                weight = weight.reshape(z.grad.shape[0], self.shape[0], z.grad.shape[1]*self.shape[1])
                z_grad = z.grad.repeat(self.shape[1], axis=1).reshape(z.grad.shape[0], 1, -1)
                sum_mask = np.tile(np.eye(self.shape[1]), z.grad.shape[1])
                accumulated_grad = ((weight * z_grad) @ sum_mask.T).sum(0)
                self._grad += accumulated_grad / z.grad.shape[0]
        
        if debug:
            print('Current Grad:', self._grad)
            print('Shape:', self._grad.shape)
            print('-'*5)
            print()

# ====================================================================================================

class Constant(Expression):

    def __init__(self, value):
        super(Constant, self).__init__(value, name=f'{value}<Const>')

    def compute_grad(self):
        self._grad = array(1)

# ====================================================================================================

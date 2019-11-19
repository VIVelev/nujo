from copy import deepcopy

import numpy as np
from numpy import array

from . import modes

__all__ = [
    'Expression',
    'Variable',
    'Constant',
]


# ====================================================================================================
# ====================================================================================================

class Expression:

    z_counter = 0

    @staticmethod
    def get_z_count():
        Expression.z_counter += 1
        return Expression.z_counter


    def __init__(self, value, name='undefined', children=[]):
        self.value = value
        self.name = name
        self.children = children

        self._dependencies = []
        self._grad = None

    @property
    def dependencies(self):
        return self._dependencies

    @dependencies.setter
    def dependencies(self, value):
        self._dependencies = value

    @property
    def grad(self):
        pass

    @property
    def T(self):
        transposed = deepcopy(self)
        transposed.value = self.value.T
        return transposed

    @property
    def shape(self):
        self.value = array(self.value)
        return self.value.shape

    def backward(self):
        _ = self.grad
        for child in self.children:
            child.backward()

        Expression.z_counter = 0

    def __add__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            return Variable(self.value + other.value, name='Add::NO_DIFF')

        z = Variable(self.value + other.value,
                        name=f'Add::Z_{Expression.get_z_count()}',
                        children=[self, other])

        self.dependencies.append(( array(1), z ))
        other.dependencies.append(( array(1), z ))

        return z

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            return Variable(self.value - other.value, name='Sub::NO_DIFF')

        z = Variable(self.value - other.value,
                        name=f'Sub::Z_{Expression.get_z_count()}',
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
            return Variable(self.value * other.value, name='Mul::NO_DIFF')

        z = Variable(self.value * other.value,
                        name=f'Mul::Z_{Expression.get_z_count()}',
                        children=[self, other])

        self.dependencies.append(( array(other.value), z ))
        other.dependencies.append(( array(self.value), z ))

        return z

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            return Variable(self.value / other.value, 'TrueDiv::NO_DIFF')

        z = Variable(self.value / other.value,
                        name=f'TrueDiv::Z_{Expression.get_z_count()}',
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
            return Variable(self.value ** other.value, name='Pow::NO_DIFF')

        z = Variable(self.value ** other.value,
                        name=f'Pow::Z_{Expression.get_z_count()}',
                        children=[self, other])

        self.dependencies.append(( array(other.value*self.value**(other.value-1)), z ))
        other.dependencies.append(( array(1), z )) # TODO: FIX (wrong partial)

        return z

    def __matmul__(self, other):
        assert self.shape[1] == other.shape[0]

        if not isinstance(other, Expression):
            other = Constant(other)

        if not modes.DIFF_ENABLED:
            return Variable(self.value @ other.value, name='MatMul::NO_DIFF')

        z = Variable(self.value @ other.value,
                        name=f'MatMul::Z_{Expression.get_z_count()}',
                        children=[self, other])

        #################### CALC MATRIX PARTIALS ####################
        # Z = XW
        # Calc:
        #   - dself = dZ/dX
        #   - dother = dZ/dW

        # ------------------------------------------------------------
        dself = np.ones((z.shape[0]*self.shape[0],
                            z.shape[1]*self.shape[1]))

        i, j = 0, 0 # indecies of Z
        l, m = 0, 0 # indecies of X
        # p, q : indecies of dZ/dX
        for p in range(dself.shape[0]):
            for q in range(dself.shape[1]):
                if l == i:
                    dself[p, q] = other.value[m, j]

                j = q % z.shape[1]
                m = q % self.shape[1]

            i = q % z.shape[0]
            l = p % self.shape[0]
        
        # ------------------------------------------------------------
        dother = np.ones((z.shape[0]*other.shape[0],
                            z.shape[1]*other.shape[1]))
        
        i, j = 0, 0 # indecies of Z
        l, m = 0, 0 # indecies of W
        # p, q : indecies of dZ/dW
        for p in range(dother.shape[0]):
            for q in range(dother.shape[1]):
                if m == j:
                    dother[p, q] = self.value[i, l]

                j = q % z.shape[1]
                m = q % other.shape[1]

            i = q % z.shape[0]
            l = p % other.shape[0]
        
        ##############################################################
        
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

    @property
    def grad(self, debug=False):

        if self._grad is None:
            if len(self.dependencies) == 0:
                self._grad = array(1)
            else:
                self._grad = 0

                if debug:
                    print()
                    print('='*30)
                    print(self, self.shape, ':: dependencies')
                for weight, z in self.dependencies:
                    
                    if debug:
                        print('-'*10)
                        print('Weight of `Z_prev Grad`:', weight)
                        print('Shape:', weight.shape)
                        print('-'*5)
                        print('Z_prev Grad:', z.grad)
                        print('Shape:', z.grad.shape)
                        print('-'*5)
                    
                    if (not weight.shape or not z.grad.shape) or \
                        (weight.shape == (1, 1) or z.grad.shape == (1, 1)):
                        self._grad += weight * z.grad

                    else:
                        z_grad = z.grad.repeat(self.shape[1], axis=1)

                        j = 0
                        for i in range(0, weight.shape[0], self.shape[0]):
                            weight[i:i+self.shape[0]] = weight[i:i+self.shape[0]] * z_grad[j]
                            if i % self.shape[0] == 0:
                                j+=1

                        mask_weight = np.tile(np.eye(self.shape[1]), weight.shape[1]//self.shape[1])
                        mask_accumulate = np.tile(np.eye(self.shape[0]), z.shape[0])
                        accumulated_grad = mask_accumulate @ (weight @ mask_weight.T)

                        self._grad = accumulated_grad / z.shape[0]

                    if debug:
                        print('Current Grad:', self._grad)
                        print('Shape:', self._grad.shape)
                        print('-'*5)
                        print()

        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    def zero_grad(self):
        self.grad = None
        self.dependencies = []

# ====================================================================================================

class Constant(Expression):

    def __init__(self, value):
        super(Constant, self).__init__(value, name=f'Const::({value})')

    @property
    def grad(self):
        if self._grad is None:
            self._grad = array(1)
        
        return self._grad

# ====================================================================================================

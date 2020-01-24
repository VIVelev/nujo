from abc import abstractmethod
from copy import deepcopy

from numpy import array

from nujo.autodiff.constant import Constant
from nujo.autodiff.functions import Addition
from nujo.autodiff.utils import counter, matrix_dotprod_differentiation


class Tensor:
    ''' Tensor - multi-dimensional array

    Tensor is the basic block of computation in Nujo.

    Parameters:
    -----------
    value : value
    name : string
    children : list

    '''

    _z_counter = counter()    # Counter used to indicate the order of computations.

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

    @abstractmethod
    def compute_grad(self):    # To be overridden by subclasses.
        pass

    def zero_grad(self):
        self.dependencies = []
        self._grad = None

    def backward(self):
        self.compute_grad()
        for child in self.children:
            child.backward()

        self._z_counter.reset()    # A new forward pass awaits.

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        return Addition(self, other)()

    def __radd__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)
        
        return other.__add__(self)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        if DIFF_ENABLED:
            suffix = '<Sub>(NO_DIFF)'
            return Variable(
                self.value - other.value,
                name=self.name + (suffix not in self.name) * suffix)

        z = Variable(self.value - other.value,
            name=f'Z_{self._z_counter.get()}<Sub>',
            children=[self, other])

        self.dependencies.append(( array(1), z ))
        other.dependencies.append(( array(-1), z ))

        return z

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        return other.__sub__(self)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        if not DIFF_ENABLED:
            suffix = '<Mul>(NO_DIFF)'
            return Variable(
                self.value * other.value,
                name=self.name + (suffix not in self.name) * suffix)

        z = Variable(self.value * other.value,
            name=f'Z_{self._z_counter.get()}<Mul>',
            children=[self, other])

        self.dependencies.append(( array(other.value), z ))
        other.dependencies.append(( array(self.value), z ))

        return z

    def __rmul__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)
        
        return other.__mul__(self)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        if not DIFF_ENABLED:
            suffix = '<TrueDiv>(NO_DIFF)'
            return Variable(
                self.value / other.value,
                name=self.name + (suffix not in self.name) * suffix)

        z = Variable(self.value / other.value,
            name=f'Z_{self._z_counter.get()}<TrueDiv>',
            children=[self, other])

        self.dependencies.append(( array(1/other.value), z ))
        other.dependencies.append(( array((-self.value)/(other.value**2)), z ))

        return z

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)
        
        return other.__truediv__(self)

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        if not DIFF_ENABLED:
            suffix = '<Pow>(NO_DIFF)'
            return Variable(
                self.value ** other.value,
                name=self.name + (suffix not in self.name) * suffix)

        z = Variable(self.value ** other.value,
            name=f'Z_{self._z_counter.get()}<Pow>',
            children=[self, other])

        self.dependencies.append(( array(other.value*self.value**(other.value-1)), z ))
        other.dependencies.append(( array(1), z )) # TODO: FIX (wrong partial)

        return z

    def __matmul__(self, other):
        assert self.shape[1] == other.shape[0]

        if not isinstance(other, Tensor):
            other = Constant(other)

        if not DIFF_ENABLED:
            suffix = '<MatMul>(NO_DIFF)'
            return Variable(
                self.value @ other.value,
                name=self.name + (suffix not in self.name) * suffix)

        z = Variable(self.value @ other.value,
            name=f'Z_{self._z_counter.get()}<MatMul>',
            children=[self, other])

        dself, dother = matrix_dotprod_differentiation(z, self.value, other.value)
        self.dependencies.append(( dself, z ))
        other.dependencies.append(( dother, z ))

        return z

    def __rmatmul__(self, other):
        if not isinstance(other, Tensor):
            other = Constant(other)

        return other.__matmul__(self)

    def __repr__(self):
        return self.name

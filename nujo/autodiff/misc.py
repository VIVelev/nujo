from numpy import ones

__all__ = [
    'counter',
    'generate_tensor_name',
    'matrix_dotprod_differentiation',
]

class counter:
    n = 0

    @classmethod
    def get(cls):
        cls.n += 1
        return cls.n

    @classmethod
    def reset(cls):
        cls.n = 0

def generate_tensor_name(i, func_name):
    return f'Z:{i}<{func_name}>'

def matrix_dotprod_differentiation(Z, X, W):
    #################### CALC MATRIX PARTIALS ####################
    # Z = XW
    # Calc:
    #   - dX = dZ/dX
    #   - dW = dZ/dW

    # ------------------------------------------------------------
    dX = ones((Z.shape[0]*X.shape[0], Z.shape[1]*X.shape[1]))

    i, j = 0, 0 # indecies of Z
    l, m = 0, 0 # indecies of X
    # p, q : indecies of dX

    for p in range(dX.shape[0]):
        for q in range(dX.shape[1]):
            if l == i:
                dX[p, q] = W[m, j]

            j = q % Z.shape[1]
            m = q % X.shape[1]

        i = q % Z.shape[0]
        l = p % X.shape[0]
    
    # ------------------------------------------------------------
    dW = ones((Z.shape[0]*W.shape[0], Z.shape[1]*W.shape[1]))
    
    i, j = 0, 0 # indecies of Z
    l, m = 0, 0 # indecies of W
    # p, q : indecies of dW

    for p in range(dW.shape[0]):
        for q in range(dW.shape[1]):
            if m == j:
                dW[p, q] = X[i, l]

            j = q % Z.shape[1]
            m = q % W.shape[1]

        i = q % Z.shape[0]
        l = p % W.shape[0]
    
    ##############################################################

    return dX, dW

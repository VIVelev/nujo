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
    return f'Z:{i}{func_name}'


def matrix_dotprod_differentiation(X, W):
    # CALC MATRIX PARTIALS
    # Z = XW
    # Calc:
    #   - dX = dZ/dX
    #   - dW = dZ/dW

    # ------------------------------------------------------------
    dX = ones((X.shape[0]**2, W.shape[1] * X.shape[1]))

    i, j = 0, 0  # indecies of Z
    k, m = 0, 0  # indecies of X
    # p, q : indecies of dX

    for p in range(dX.shape[0]):
        for q in range(dX.shape[1]):
            if k == i:
                dX[p, q] = W[m, j]

            j = q % W.shape[1]
            m = q % X.shape[1]

        i = q % X.shape[0]
        k = p % X.shape[0]

    # ------------------------------------------------------------
    dW = ones((X.shape[0] * W.shape[0], W.shape[1]**2))

    i, j = 0, 0  # indecies of Z
    k, m = 0, 0  # indecies of W
    # p, q : indecies of dW

    for p in range(dW.shape[0]):
        for q in range(dW.shape[1]):
            if m == j:
                dW[p, q] = X[i, k]

            j = q % W.shape[1]
            m = q % W.shape[1]

        i = q % X.shape[0]
        k = p % W.shape[0]

    ##############################################################

    return dX, dW

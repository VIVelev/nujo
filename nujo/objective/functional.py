from nujo.math.aggregate import mean
from nujo.math.scalar import abs

__all__ = [
    'mean_abs_err',
    'mean_sq_err',
]


def mean_abs_err(input, target, dim=None, keepdim=False):
    return mean(abs(input - target), dim=dim, keepdim=keepdim)


def mean_sq_err(input, target, dim=None, keepdim=False):
    return mean((input - target)**2, dim=dim, keepdim=keepdim)

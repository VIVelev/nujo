from numpy import mean as np_mean

from nujo.autodiff import Tensor

__all__ = [
    'nj_mean',
]


def nj_mean(*args: Tensor, dim: int = None, keepdim=False) -> Tensor:
    if len(args) > 1:
        return np_mean(args, axis=dim, keepdims=keepdim)

    else:
        return Tensor(np_mean(args[0].value, axis=dim, keepdims=keepdim),
                      creator=args[0].creator)

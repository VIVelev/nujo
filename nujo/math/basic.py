from numpy import sum as np_sum

from nujo.autodiff import Tensor

__all__ = [
    'nj_sum',
]


def nj_sum(*args: Tensor, dim: int = None, keepdim=False) -> Tensor:
    if len(args) > 1:
        return np_sum(args, axis=dim, keepdims=keepdim)

    else:
        return Tensor(np_sum(args[0].value, axis=dim, keepdims=keepdim),
                      creator=args[0].creator)

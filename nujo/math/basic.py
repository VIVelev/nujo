from nujo.autodiff import Tensor

__all__ = [
    '_sum',
]


def _sum(*args: Tensor) -> Tensor:
    if len(args) > 1:
        return sum(args)

    else:
        return Tensor(sum(args[0]), creator=args[0].creator)

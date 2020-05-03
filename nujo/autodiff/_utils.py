__all__ = [
    '_if_not_none',
]


def _if_not_none(*args) -> list:
    return [arg for arg in args if arg is not None]

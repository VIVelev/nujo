__all__ = [
    'counter',
    'if_not_none',
]


class counter:
    n = 0

    @classmethod
    def get(cls) -> int:
        cls.n += 1
        return cls.n

    @classmethod
    def reset(cls) -> None:
        cls.n = 0


def if_not_none(*args) -> list:
    return [arg for arg in args if arg is not None]

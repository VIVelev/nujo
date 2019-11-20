class counter:
    n = 0

    @classmethod
    def get(cls):
        cls.n += 1
        return cls.n

    @classmethod
    def reset(cls):
        cls.n = 0

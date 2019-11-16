__all__ = [
    'GRAD_ENABLED',
    'no_grad',
]


GRAD_ENABLED = True

class no_grad():

    def __enter__(self):
        global GRAD_ENABLED
        GRAD_ENABLED = False

    def __exit__(self, type, value, traceback):
        global GRAD_ENABLED
        GRAD_ENABLED = True

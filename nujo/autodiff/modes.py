__all__ = [
    'DIFF_ENABLED',
    'no_diff',
]

DIFF_ENABLED = True


class no_diff():
    ''' No Differentiation block

    Creates a block of code where no differentiation is done.

    '''
    def __enter__(self):
        global DIFF_ENABLED
        DIFF_ENABLED = False

    def __exit__(self, type, value, traceback):
        global DIFF_ENABLED
        DIFF_ENABLED = True

__all__ = [
    'DIFF_ENABLED',
    'no_diff',
]

DIFF_ENABLED = True
''' This variable controls whether nujo to compute gradients
for the tensors in the computation graph:
    - True = differentiation enabled, compute gradients
      for the diff enabled (diff=True) tensors.
    - False = differentiation disabled, do NOT compute gradients.

Another way to see it is:
 - if DIFF_ENABLED is True, the computation graph is updated,
 otherwise it is not.
'''


class no_diff():
    ''' No Differentiation block

    Creates a block of code where no differentiation is done.
    a.k.a. No gradients are computed for whatever tensor.

    '''
    def __enter__(self):
        global DIFF_ENABLED
        DIFF_ENABLED = False

    def __exit__(self, type, value, traceback):
        global DIFF_ENABLED
        DIFF_ENABLED = True

from h5py import File, Group


class H5Dict(object):
    '''

    Parameters:
    -----------
     :

    Returns:
    --------
     :
    '''
    def __init__(self, path, mode='a'):
        self.data = None
        self.path = path

    @staticmethod
    def is_supported_type(path):
        return (isinstance(path, dict) or isinstance(path, Group)
                or isinstance(path, File))


def save_to_binary_h5py():
    pass

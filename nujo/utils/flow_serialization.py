from numpy import array, load, save

from nujo.flow import Flow

__all__ = ['save_flow', 'load_flow']


def save_flow(flow: Flow) -> None:
    '''Save flow

    Saves the parameters of a Flow to a file named:\n
    *name of flow*_parameters.npy in the current directory.

    Parameters:
    -----------
    Flow : will take parameters and save them to a .npy file

    '''
    name = flow.name
    params = array([var for var in flow.parameters()])
    save(f'{name}_parameters', params, allow_pickle=True)


def load_flow(filename: str) -> Flow:
    '''Load flow

    Load the parameters of a Flow from a file

    Parameters:
    -----------
    - Flow : this flow will have its parameters set to the read ones
    - filename : the file path to get array from

    '''
    return Flow(_chain=load(filename, allow_pickle=True))

from numpy import array, load, save

from nujo.flow import Flow

__all__ = ['save_flow', 'load_flow']


def save_flow(flow: Flow) -> None:
    '''Save flow

    Saves the parameters of a network to a file named:\n
    *name of flow*_parameters.npy in the current directory.

    Parameters:
    -----------
    Flow : will take parameters and save them to a .npy file


    '''
    name = flow.name
    params = array(flow.parameters)
    save(f'{name}_parameters', params, allow_pickle=True)


def load_flow(filename: str) -> Flow:
    '''Load flow

    Load the parameters as an array of a network from a file

    Parameters:
    -----------
    - Flow : this flow will have its parameters set to the read ones
    - filename : the file path to get array from

    '''
    flow = Flow()
    flow.parameters = load(filename, allow_pickle=True).tolist()
    return flow

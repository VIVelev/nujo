from numpy import array, save, load

from nujo.flow import Flow


def saveNetwork():
    '''Save network

    Saves the parameters of a network to a file named:\n
    *name of flow*_parameters.npy in the current directory.

    Parameters:
    -----------
    Flow : will take parameters and save them to a .npy file


    '''
    def __init__(self, flow: Flow) -> None:
        self.name = flow.name
        self.params = array(flow.params)
        save(f'{self.name}_parameters', self.params)


def loadNetwork():
    '''Load network

    Load the parameters as an array of a network from a file

    Parameters:
    -----------
     - Flow : this flow will have its parameters set to the read ones
     - filename : the file path to get array from

    '''
    def __init__(self, flow: Flow, filename: str):
        load(filename, flow.parameters)

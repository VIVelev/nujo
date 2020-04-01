from numpy import array, save, load

from nujo.flow import Flow


def save_flow(self, flow: Flow) -> None:
    '''Save flow

    Saves the parameters of a network to a file named:\n
    *name of flow*_parameters.npy in the current directory.

    Parameters:
    -----------
    Flow : will take parameters and save them to a .npy file


    '''
    self.name = flow.name
    self.params = array(flow.params)
    save(f'{self.name}_parameters', self.params)


def load_flow(self, flow: Flow, filename: str):
    '''Load flow

    Load the parameters as an array of a network from a file

    Parameters:
    -----------
    - Flow : this flow will have its parameters set to the read ones
    - filename : the file path to get array from

    '''
    load(filename, flow.parameters)

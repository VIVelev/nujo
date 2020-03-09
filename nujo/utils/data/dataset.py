from numpy import array, empty, float32, vstack

from nujo.utils.data.dataset_iterator import DatasetIterator
from nujo.utils.data.nujo_dir import HOME_DIR


class Dataset:
    '''

    Parameters:
    -----------
    name : will be downloaded from the UCI ML repo

    Returns:
    --------
    tuple : stores the csv dataset,
    - np array with floating point integers
    - np array with labels
    '''
    def __init__(self, name):
        # self.name = expanduser('~/.nujo/') + name
        if '.data' not in name:
            self.name += '.data'

    def _load_from_file(self):
        with open(self.name, 'r+') as data:
            lines = data.readlines()
            self.X = empty((0, 4))
            self.y = empty((0, 1))

        for line in lines[:-1]:
            x = array([line.split(',')[:4]], dtype=float32)
            self.X = vstack((self.X, x))
            y = array([line.split(',')[-1][:-1]], dtype='U')
            self.y = vstack((self.y, y))

    def __iter__(self):
        return DatasetIterator(self)


if __name__ == '__main__':
    data = Dataset('iris')
    for line in data:
        print(line)

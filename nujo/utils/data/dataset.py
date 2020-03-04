from numpy import array, empty, float32, vstack
from nujo.utils.data.dataset_iterator import DatasetIterator


class Dataset:
    def __init__(self, name):
        file = '~/.nujo/' + name
        if not '.data' in name:
            file += '.data'
        with open(file) as data:
            lines = data.readlines()
            self.X = empty((0, 4))
            self.y = empty((0, 1))

            for line in lines[:-1]:
                x = array([line.split(',')[:4]], dtype=float32)
                self.X = vstack((self.X, x))
                y = array([line.split(',')[-1]])
                self.y = vstack((self.y, y))

    def __iter__(self):
        return DatasetIterator(self)

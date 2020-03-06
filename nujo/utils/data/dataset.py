from numpy import array, empty, float32, vstack
from nujo.utils.data.dataset_iterator import DatasetIterator
from os.path import expanduser


class Dataset:
    def __init__(self, name):
        file = expanduser('~/.nujo/') + name
        if '.data' not in name:
            file += '.data'
        with open(file, 'r+') as data:
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


data = Dataset('iris')
for line in data:
    print(line)

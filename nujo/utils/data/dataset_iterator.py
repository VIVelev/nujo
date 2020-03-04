
class DatasetIterator:
    ''' Dataset Iterator

    Parameters:
    -----------
    dataset : ....

    '''

    def __init__(self, dataset):
        self._data = dataset
        self._index = 0

    def __next__(self):
        if self._index >= len(self._data.X):
            raise StopIteration

        result = (self._data.X[self._index], self._data.Y[self._index])
        self._index += 1
        return result

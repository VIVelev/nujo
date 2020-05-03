from nujo.utils.data.dataset import Dataset


class DatasetIterator:
    ''' Dataset Iterator

    Parameters:
    -----------
     - dataset: Dataset, the dataset to iterate over

    Returns:
    --------
     - next element : numpy array of values with the label

    '''
    def __init__(self, dataset: Dataset):
        self._data = dataset
        self._index = 0

    def __next__(self):
        if self._index >= len(self._data.X):
            raise StopIteration

        result = (self._data.X[self._index], self._data.y[self._index])
        self._index += 1

        return result

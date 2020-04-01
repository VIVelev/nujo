from os import mkdir
from os.path import exists

from numpy import array, empty, vstack
from requests import get

from nujo.utils.data.nujo_dir import HOME_DIR


class DatasetLoader:
    '''

    Parameters:
    -----------
    name : will be downloaded from the UCI ML repo
    '''
    _UCI_REPO_URL = 'archive.ics.uci.edu/ml/machine-learning-databases/{}/{}'

    def __init__(self, dataset):
        self.name = dataset.name  # with .data
        self._link = self._UCI_REPO_URL.format(self.name[:-4], self.name)
        with open(self.name, 'r+') as data:
            lines = data.readlines()

        dataset.X = empty((0, len(lines[0].split(','))))
        # number of columns
        for line in lines[:-1]:  # last row is \n
            x = array(line.strip().split(','))
            dataset.X = vstack((dataset.X, x))

    def download(self) -> None:
        r = get(self._link)
        file = f'{HOME_DIR}{self.name}.data'
        if not exists(HOME_DIR):
            mkdir(HOME_DIR)
            print('Directory "~/.nujo" Created ')
        else:
            print('Directory "~/.nujo" already exists')
        print(f'File {self.name} has been created.')
        with open(file) as f:
            f.write(r.content)


if __name__ == '__main__':
    DatasetLoader('iris').download()

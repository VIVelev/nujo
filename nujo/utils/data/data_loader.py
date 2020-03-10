from os import mkdir
from os.path import exists

from nujo.utils.data.constants import HOME_DIR
from requests import get


class DataLoader:
    '''

    Parameters:
    -----------
    name : will be downloaded from the UCI ML repo
    '''
    _UCI_REPO_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/{}/{}.data'

    def __init__(self, name):
        self.name = name
        self._link = self._UCI_REPO_URL.format(self.name)

    def download(self):
        r = get(self._link)
        file = HOME_DIR + self.name + '.data'
        if not exists(HOME_DIR):
            mkdir(HOME_DIR)
            print('Directory "~/.nujo" Created ')
        else:
            print('Directory "~/.nujo" already exists')
        print(f'File {self.name} has been created.')
        with open(file) as f:
            f.write(r.content)


if __name__ == '__main__':
    DataLoader('iris').download()

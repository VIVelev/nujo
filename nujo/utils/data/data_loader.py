from os import mkdir
from os.path import exists, expanduser

import requests


class DataLoader:
    '''



    Parameters:
    -----------
    name : will be downloaded from the UCI ML repo
    '''
    _UCI_REPO_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/{}/{}.data'
    _HOME_DIR = expanduser('~/.nujo')

    def __init__(self, name):
        self._name = name
        self._line = _UCI_REPO_URL.format(self._name)

    def download(self):
        r = requests.get(self._link)
        file = self._HOME_DIR + self._name + '.data'
        if not exists(self._HOME_DIR):
            mkdir(self._HOME_DIR)
            print('Directory "~/.nujo" Created ')
        else:
            print('Directory "~/.nujo" already exists')
        print(f'File {self._name} has been created.')
        with open(file) as f:
            f.write(r.content)


DataLoader('iris').download()

from os import mkdir
from os.path import exists, expanduser
import requests


class DataLoader:
    def __init__(self, name):
        self._name = name
        self._link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/{0}/{0}.data'.format(
            name)

    def download(self):
        r = requests.get(self._link)
        file = expanduser('~/.nujo/') + self._name + '.data'
        nujo = expanduser('~/.nujo')
        if not exists(nujo):
            mkdir(nujo)
            print("Directory '~/.nujo' Created ")
        else:
            print("Directory '~/.nujo' already exists")
        print('File {} has been created.'.format(self.name))
        open(file, 'wb').write(r.content)


data = DataLoader('iris').download()

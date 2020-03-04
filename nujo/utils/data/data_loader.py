import requests


class DataLoader:
    def __init__(self, name):
        self._name = name
        self._link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/{0}/{0}.data'.format(
            name)

    def download(self):
        r = requests.get(self._link)
        file = '~/.nujo/' + self._name + '.data'
        open(file, 'wb').write(r.content)

''' Documentation generation and update procedure
'''

from os import listdir, system
from os.path import dirname, isdir, join
from shutil import move, rmtree


def gen_docs():
    print('Generating HTML documentation for nujo out of docstrings...')

    nujo = '../nujo'
    # nujo = join(dirname(__file__), '../nujo')
    docs = dirname(__file__)

    system(f'pdoc3 --html {nujo} --output-dir {docs} --force')
    print('--Done--')


def extract_docs():
    print('Extracting documentation...')

    # source = join(dirname(__file__), 'nujo')
    source = 'nujo'
    dest = dirname(__file__)

    for f in listdir(source):
        print(f)
        current_dest = dest + '/' + f
        if isdir(current_dest):
            rmtree(current_dest)

        move(source + '/' + f, current_dest)

    rmtree(source)
    print('--Done--')


if __name__ == '__main__':
    gen_docs()
    extract_docs()

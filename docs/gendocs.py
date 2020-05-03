''' Documentation generation and update procedure
'''

from os import listdir, system
from os.path import dirname, isdir
from shutil import move, rmtree
from sys import argv


def gen_docs(flag):
    print('Generating HTML documentation for nujo out of docstrings...')

    nujo = '../nujo'
    if (flag):
        from os.path import join
        nujo = join(dirname(__file__), '../nujo')

    docs = dirname(__file__)

    system(f'pdoc3 --html {nujo} --output-dir {docs} --force')
    print('--Done--')


def extract_docs(flag):
    print('Extracting documentation...')
    print(__file__)

    source = 'nujo'
    if (flag):
        from os.path import join
        source = join(dirname(__file__), 'nujo')
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
    flag = True
    if len(argv) != 1:
        flag = (argv[1].strip().lower() == 'true')
    gen_docs(flag)
    extract_docs(flag)

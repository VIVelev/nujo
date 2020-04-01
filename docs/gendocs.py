''' Documentation generation and update procedure
'''

import os
import shutil


def gen_docs():
    print('Generating HTML documentation for nujo out of docstrings...')

    nujo = os.path.join(os.path.dirname(__file__), '../nujo')
    docs = (os.path.dirname(__file__)

    os.system(f'pdoc3 --html {nujo} --output-dir {docs} --force')
    print('Done.\n')


def extract_docs():
    print('Extracting documentation...')

    source = os.path.join(os.path.dirname(__file__), 'nujo')
    dest = os.path.dirname(__file__)

    for f in os.listdir(source):
        current_dest = dest + '/' + f
        if os.path.isdir(current_dest):
            shutil.rmtree(current_dest)

        shutil.move(source + '/' + f, current_dest)

    shutil.rmtree(source)
    print('Done.\n')


if __name__ == '__main__':
    gen_docs()
    extract_docs()

import os
import sys

if __name__ == '__main__':
    path_to_script_with_args = ' '.join(sys.argv[1:])

    os.system(
        f'python -m cProfile -o output.pstats {path_to_script_with_args}')
    os.system(f'gprof2dot --colour-nodes-by-selftime -f pstats output.pstats |\
             dot -Tpng -o output.png')

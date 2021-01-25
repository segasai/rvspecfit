import os

import pathlib
path = str(pathlib.Path(__file__).parent.absolute())


def test_scripts():
    tmp = os.system(path + '/make_templ.sh')
    assert (tmp == 0)
    tmp = os.system(path + '/gen_test_templ_grid.sh')
    assert (tmp == 0)
    tmp = os.system(path + '/gen_test_templ.sh')
    assert (tmp == 0)

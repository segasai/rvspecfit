import os


def test_scripts():
    tmp = os.system('./make_templ.sh')
    assert (tmp == 0)
    tmp = os.system('./gen_test_templ_grid.sh')
    assert (tmp == 0)
    tmp = os.system('./gen_test_templ.sh')
    assert (tmp == 0)

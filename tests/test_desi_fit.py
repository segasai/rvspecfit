import pathlib
from test_01template_creation import run_script

path = str(pathlib.Path(__file__).parent.absolute())


def test_scripts():
    run_script(path + '/scripts/desi_fit.sh')

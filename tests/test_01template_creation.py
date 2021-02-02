import os
import subprocess
import pathlib
import rvspecfit.make_nd
import rvspecfit.read_grid
import rvspecfit.make_ccf
import rvspecfit.make_interpol

path = str(pathlib.Path(__file__).parent.absolute())


def run_script(script):
    pipe = subprocess.Popen(script, stdout=subprocess.PIPE)
    stdout = pipe.stdout.read().decode()
    for l in stdout.split('\n'):
        args = l.split(' ')
        script_name = args[0].split('/')[-1]
        if script_name == 'rvs_make_nd':
            rvspecfit.make_nd.main(args[1:])
        if script_name == 'rvs_make_interpol':
            rvspecfit.make_interpol.main(args[1:])
        if script_name == 'rvs_make_ccf':
            rvspecfit.make_ccf.main(args[1:])
        if script_name == 'rvs_read_grid':
            rvspecfit.read_grid.main(args[1:])


def test_scripts():
    run_script(path + '/make_templ.sh')
    run_script(path + '/gen_test_templ_grid.sh')
    run_script(path + '/gen_test_templ.sh')

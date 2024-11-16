import subprocess
import pathlib
import rvspecfit.make_nd
import rvspecfit.read_grid
import rvspecfit.make_ccf
import rvspecfit.make_interpol
import rvspecfit.regularize_grid
import rvspecfit.nn.train_interpolator
import rvspecfit.desi.desi_fit

path = str(pathlib.Path(__file__).parent.absolute())

# the goal of this file is to convert regular bash script into
# python calls where we could get coverage


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
        if script_name == 'rvs_regularize_grid':
            rvspecfit.regularize_grid.main(args[1:])
        if script_name == 'rvs_train_nn_interpolator':
            rvspecfit.nn.train_interpolator.main(args[1:])
        if script_name == 'rvs_desi_fit':
            rvspecfit.desi.desi_fit.main(args[1:])


def test_scripts_nn():
    run_script(path + '/scripts/make_templ_nn.sh')


def test_scripts_triang():
    run_script(path + '/scripts/make_templ_aat_triang.sh')


def test_scripts_regul():
    run_script(path + '/scripts/make_templ_regul.sh')


def test_scripts_templ():
    run_script(path + '/scripts/gen_test_templ.sh')


def test_scripts_templ_grid():
    run_script(path + '/scripts/gen_test_templ_grid.sh')

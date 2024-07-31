import os

os.environ['OMP_NUM_THREADS'] = '1'
from rvspecfit import spec_fit
from rvspecfit import spec_inter
from rvspecfit import utils
import pathlib

path = str(pathlib.Path(__file__).parent.absolute())


def test_interp():
    conf = utils.read_config(path + '/yamls/config_sdss.yaml')
    interp = spec_inter.getInterpolator('sdss1', conf)
    interp.eval({'teff': 5000, 'logg': 1, 'feh': -1, 'alpha': 0.3})


if __name__ == '__main__':
    test_interp()

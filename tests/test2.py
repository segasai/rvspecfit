import os
os.environ['OMP_NUM_THREADS']='1'
import sys
from rvspecfit import spec_fit
from rvspecfit import spec_inter
from rvspecfit import utils

conf = utils.read_config('config.yaml')
interp = spec_inter.getInterpolator('sdss1', conf)
interp.eval({'teff': 5000, 'logg': 1, 'feh': -1, 'alpha': 0.3})

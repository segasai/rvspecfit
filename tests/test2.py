import sys
sys.path.append('../')
import spec_fit
import spec_inter
import utils

conf = utils.read_config('config.yaml')
interp = spec_inter.getInterpolator('sdss', conf)
interp.eval({'teff': 5000, 'logg': 1, 'feh': -1, 'alpha': 0.3})

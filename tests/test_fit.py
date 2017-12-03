import sys
sys.path.append('../')
import vel_fit
import spec_fit
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import utils
config = utils.read_config()

# read data
dat = pyfits.getdata('../examples/spec-0266-51602-0031.fits')
err = dat['ivar']
err = 1. / err**.5
err[~np.isfinite(err)] = 1e40

# construct specdata object
specdata = [spec_fit.SpecData('sdss1', 10**dat['loglam'],
                              dat['flux'], err)]
options = {'npoly': 15}
paramDict0 = {'logg': 2, 'teff': 5000, 'feh': -1, 'alpha': 0.2, 'vsini': 19}
fixParam = ['vsini']
#vel_fit.firstguess(specdata, options=options, config=config)
paramDict0 = {'logg': 2, 'teff': 5000, 'feh': -1, 'alpha': 0.2, 'vsini': 19}
fixParam = ['vsini']
res = vel_fit.doit(specdata, paramDict0, fixParam=fixParam,
                   config=config, options=options)

paramDict0 = {'logg': 2, 'teff': 5000, 'feh': -1, 'alpha': 0.2, 'vsini': 19}
fixParam = []
res = vel_fit.doit(specdata, paramDict0, fixParam=fixParam,
                   config=config, options=options)

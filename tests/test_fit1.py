import os
os.environ['OMP_NUM_THREADS']='1'
import sys
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt

from rvspecfit import fitter_ccf
from rvspecfit import vel_fit
from rvspecfit import spec_fit
from rvspecfit import utils

config = utils.read_config()

# read data
dat = pyfits.getdata('./spec-0266-51602-0031.fits')
err = dat['ivar']
err = 1. / err**.5
err[~np.isfinite(err)] = 1e40

# construct specdata object
specdata = [spec_fit.SpecData('sdss1', 10**dat['loglam'],
                              dat['flux'], err)]
options = {'npoly': 15}
res = fitter_ccf.fit(specdata, config)
paramDict0 = res['best_par']
fixParam = []
if res['best_vsini'] is not None:
    paramDict0['vsini'] = res['best_vsini']
res1 = vel_fit.process(specdata, paramDict0, fixParam=fixParam,
                    config=config, options=options)
print (res1)
plt.figure(figsize=(6, 2), dpi=300)
plt.plot(specdata[0].lam, specdata[0].spec, 'k-')
plt.plot(specdata[0].lam, res1['yfit'][0], 'r-')
plt.tight_layout()
plt.savefig('test_fit1.png')

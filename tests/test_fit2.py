import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
from rvspecfit import utils
from rvspecfit import vel_fit
from rvspecfit import spec_fit

config = utils.read_config('test.yaml')

# read data
lamcen=5000.
lamcen_air= lamcen / (
                1.0 + 2.735182E-4 + 131.4182 / lamcen**2 + 2.76249E8 / lamcen**4)
v0 = np.random.normal(0,100)
lamcen1 = lamcen * (1+v0/3e5)
resol=1000.
w=lamcen/resol/2.35
lam = np.linspace(4600,5400,800)
spec0 = 1-0.02*np.exp(-0.5*((lam-lamcen1)/w)**2)
espec=spec0 * 0.001
spec =np.random.normal(spec0,espec)
# construct specdata object
specdata = [spec_fit.SpecData('test', lam,spec, espec)]
options = {'npoly': 15}
paramDict0 = {'logg': 2, 'teff': 5000, 'feh': -0, 'alpha': 0.2, 'vsini': 0.1}
fixParam = []#'vsini']
res = vel_fit.process(
    specdata, paramDict0, fixParam=fixParam, config=config, options=options)

#res = vel_fit.process(
#    specdata, paramDict0, fixParam=fixParam, config=config, options=options)
print (res['vel']-v0,res['vel_err'])
if False:
    plt.figure(figsize=(6, 2), dpi=300)
    plt.plot(specdata[0].lam, specdata[0].spec, 'k-')
    plt.plot(specdata[0].lam, res['yfit'][0], 'r-')
    plt.tight_layout()
    plt.savefig('test_fit2.png')

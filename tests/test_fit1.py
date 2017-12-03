import sys
sys.path.append('../')
import vel_fit
import spec_fit
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import fitter_ccf
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
res = fitter_ccf.doitquick(specdata, config)
paramDict0 = res['bestpar']
fixParam = []
if res['bestvsini'] is not None:
    paramDict0['vsini'] = res['bestvsini']
res1 = vel_fit.doit(specdata, paramDict0, fixParam=fixParam,
                    config=config, options=options)
plt.figure(figsize=(6,2),dpi=300)
plt.plot(specdata[0].lam,specdata[0].spec,'k-')
plt.plot(specdata[0].lam,res1['yfit'][0],'r-')
plt.tight_layout()
plt.savefig('test_fit1.png')
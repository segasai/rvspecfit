import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
from rvspecfit import utils
from rvspecfit import vel_fit
from rvspecfit import spec_fit
import mktemps

# this is the accuracy testing suite


def doone(seed, sn=100, nlam=400, doplot=False):

    np.random.seed(seed)
    config = utils.read_config('test.yaml')

    # read data
    lamcen = 5000.
    #lamcen_air = lamcen / (1.0 + 2.735182E-4 + 131.4182 / lamcen**2 +
    #                       2.76249E8 / lamcen**4)
    v0 = np.random.normal(0, 300)
    slope = (np.random.uniform(-2, 2))
    resol = 1000.
    wresol = (lamcen / resol / 2.35)
    lam = np.linspace(4600, 5400, nlam)
    #spec0 = (1 - 0.02 * np.exp(-0.5 * ((lam - lamcen1) / w)**2))*lam**slope
    teff = np.random.uniform(3000, 12000)
    feh = np.random.uniform(-2, 0)
    alpha = np.random.uniform(0, 1)
    logg = np.random.uniform(0, 5)
    c = 299792.458
    lam1 = lam / np.sqrt((1 + v0 / c) / (1 - v0 / c))
    spec0 = mktemps.getspec(lam1, teff, logg, alpha, feh,
                            wresol=wresol) * lam**slope
    spec0 = spec0 / np.median(spec0) * 10**np.random.uniform(-3, 3)
    espec = spec0 / sn
    spec = np.random.normal(spec0, espec)
    # construct specdata object
    specdata = [spec_fit.SpecData('test1', lam, spec, espec)]
    options = {'npoly': 10}
    paramDict0 = {'logg': 2.5, 'teff': 5000, 'feh': -1, 'alpha': 0.5}
    #, 'vsini': 0.0}
    #    fixParam = ['vsini']
    res = vel_fit.process(
        specdata,
        paramDict0,
        #                  fixParam=fixParam,
        config=config,
        options=options)

    ret = (v0, res['vel'], res['vel_err'])
    if doplot:
        plt.figure(figsize=(6, 2), dpi=300)
        plt.plot(specdata[0].lam, specdata[0].spec, 'k-')
        plt.plot(specdata[0].lam, res['yfit'][0], 'r-')
        plt.tight_layout()
        plt.savefig('accuracy_test.png')
        #1/0
    return ret


if __name__ == '__main__':
    if len(sys.argv) > 1:
        seed = (int(sys.argv[1]))
    else:
        seed = 1
    if len(sys.argv) > 2:
        sn = int(sys.argv[2])
    else:
        sn = 100
    ret = doone(seed, sn, doplot=True)
    print(ret)

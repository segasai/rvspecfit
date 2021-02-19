import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import astropy.io.fits as pyfits
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from rvspecfit import utils
from rvspecfit import vel_fit
from rvspecfit import spec_fit
from rvspecfit import fitter_ccf
path = str(pathlib.Path(__file__).parent.absolute())


def test_fit():
    config = utils.read_config(path + '/config.yaml')

    # read data
    dat = pyfits.getdata(path + '/data/spec-0266-51602-0031.fits')
    err = dat['ivar']
    err = 1. / err**.5
    err[~np.isfinite(err)] = 1e40

    # construct specdata object
    specdata = [
        spec_fit.SpecData('sdss1', 10**dat['loglam'], dat['flux'], err)
    ]
    options = {'npoly': 15}
    paramDict0 = {
        'logg': 2,
        'teff': 5000,
        'feh': -1,
        'alpha': 0.2,
        'vsini': 19
    }
    fixParam = ['vsini']

    paramDict0 = {
        'logg': 2,
        'teff': 5000,
        'feh': -1,
        'alpha': 0.2,
        'vsini': 19
    }
    fixParam = ['vsini']

    # fit with fixed vssini
    res = vel_fit.process(specdata,
                          paramDict0,
                          fixParam=fixParam,
                          config=config,
                          options=options)

    paramDict0 = {
        'logg': 2,
        'teff': 5000,
        'feh': -1,
        'alpha': 0.2,
        'vsini': 19
    }

    # fit witout fixin
    fixParam = []
    res = vel_fit.process(specdata,
                          paramDict0,
                          fixParam=fixParam,
                          config=config,
                          options=options)

    options = {'npoly': 15}

    # first guess fit
    xres0 = vel_fit.firstguess(specdata, config=config)

    # ccf
    res = fitter_ccf.fit(specdata, config)
    paramDict0 = res['best_par']
    fixParam = []
    if res['best_vsini'] is not None:
        paramDict0['vsini'] = res['best_vsini']
    res1 = vel_fit.process(specdata,
                           paramDict0,
                           fixParam=fixParam,
                           config=config,
                           options=options)
    print(res1)
    plt.figure(figsize=(6, 2), dpi=300)
    plt.plot(specdata[0].lam, specdata[0].spec, 'k-')
    plt.plot(specdata[0].lam, res1['yfit'][0], 'r-')
    plt.tight_layout()
    plt.savefig(path + '/plot_test_fit_sdss.png')

    # test priors
    res2 = vel_fit.process(specdata,
                           paramDict0,
                           fixParam=fixParam,
                           config=config,
                           options=options,
                           priors={'teff': (9000, 50)})
    # test rbf
    options['rbf_continuum'] = False
    res2 = vel_fit.process(specdata,
                           paramDict0,
                           fixParam=fixParam,
                           config=config,
                           options=options)

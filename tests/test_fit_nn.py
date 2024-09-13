import os

os.environ['OMP_NUM_THREADS'] = '1'
import sys
import pytest
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
    config = utils.read_config(path + '/yamls/config_nn.yaml')

    # read data
    npix = 1000
    wave = np.linspace(4000, 5000, npix)
    dat0 = wave * 0 + 1
    err = np.ones(npix) * 0.1
    rstate = np.random.default_rng(400)
    dat = rstate.normal(dat0, err)
    # construct specdata object
    specdata = [spec_fit.SpecData('aat_580v', wave, dat, err)]
    options = {'npoly': 5}
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

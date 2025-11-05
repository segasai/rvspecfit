import os

os.environ['OMP_NUM_THREADS'] = '1'
import astropy.io.fits as pyfits
import numpy as np
import sys
import time
import pathlib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rvspecfit import spec_fit
from rvspecfit import vel_fit
from rvspecfit import utils

path = str(pathlib.Path(__file__).parent.absolute())


def test_fits():
    config = utils.read_config(path + '/yamls/config_sdss.yaml')

    # read data
    dat = pyfits.getdata(path + '/data/spec-0266-51602-0031.fits')
    err = dat['ivar']
    err = 1. / err**.5
    err[~np.isfinite(err)] = 1e30

    # construct specdata object
    specdata = [
        spec_fit.SpecData('sdss1', 10**dat['loglam'], dat['flux'], err)
    ]
    rot_params = None
    resol_params = None

    params_list = [[4000, 3, -1, 0], [5000, 3, -1, 0], [6000, 2, -2, 0],
                   [5500, 5, 0, 0]]
    vel_grid = np.linspace(-600, 600, 1000)
    options = {'npoly': 10}

    t1 = time.time()
    res = spec_fit.find_best(specdata,
                             vel_grid,
                             params_list,
                             rot_params=rot_params,
                             resol_params=resol_params,
                             options=options,
                             config=config)
    t2 = time.time()
    bestv, bestpar, bestchi, vel_err = [
        res[_] for _ in ['best_vel', 'best_param', 'best_chi', 'vel_err']
    ]
    assert (np.abs(bestv - 15) < 15)
    param0 = vel_fit.firstguess(specdata, options=options, config=config)
    resfull = vel_fit.process(specdata,
                              param0,
                              resolParams=resol_params,
                              options=options,
                              config=config)
    chisquare = np.mean(
        ((specdata[0].spec - resfull['yfit'][0]) / specdata[0].espec)**2)
    assert (chisquare < 1.2)
    assert (np.abs(resfull['vel'] - 6) < 10)
    rot_params = (300, )
    plt.clf()
    ret = spec_fit.get_chisq(specdata,
                             bestv,
                             bestpar,
                             rot_params=rot_params,
                             resol_params=resol_params,
                             options=options,
                             config=config,
                             full_output=True)
    plt.plot(specdata[0].lam, specdata[0].spec, 'k')
    plt.plot(specdata[0].lam, ret['models'][0], 'r')
    plt.savefig(path + '/plot_sdss_test1.png')

    # Test the fit with the resolution matrix
    rot_params = None
    resol_mat = spec_fit.construct_resol_mat(specdata[0].lam, 50)
    resol_params = {'sdss1': resol_mat}
    ret = spec_fit.get_chisq(specdata,
                             bestv,
                             bestpar,
                             rot_params,
                             resol_params=resol_params,
                             options=options,
                             config=config,
                             full_output=True)
    plt.clf()
    plt.plot(specdata[0].lam, specdata[0].spec, 'k')
    plt.plot(specdata[0].lam, ret['models'][0], 'r')
    plt.savefig(path + '/plot_sdss_test2.png')
    # fit again with resolParams
    resfull = vel_fit.process(specdata,
                              param0,
                              resolParams=resol_params,
                              options=options,
                              config=config)
    resol_mat = spec_fit.construct_resol_mat(specdata[0].lam, 50)
    specdata = [
        spec_fit.SpecData('sdss1',
                          10**dat['loglam'],
                          dat['flux'],
                          err,
                          resolution=resol_mat)
    ]

    ret = spec_fit.get_chisq(specdata,
                             bestv,
                             bestpar,
                             rot_params,
                             options=options,
                             config=config,
                             full_output=True)
    plt.clf()
    plt.plot(specdata[0].lam, specdata[0].spec, 'k')
    plt.plot(specdata[0].lam, ret['models'][0], 'r')
    plt.savefig(path + '/plot_sdss_test3.png')
    ret = spec_fit.get_chisq_continuum(specdata, options=options)

    ret0 = spec_fit.get_chisq(
        specdata,
        bestv,
        bestpar,
        rot_params=rot_params,
        options=options,
        config=config,
    )
    ret1 = spec_fit.get_chisq(
        specdata,
        bestv,
        bestpar,
        rot_params=rot_params,
        options=options,
        config=config,
        espec_systematic={'sdss1': specdata[0].spec * 0.01})
    ret1 = spec_fit.get_chisq(specdata,
                              bestv,
                              bestpar,
                              rot_params=rot_params,
                              options=options,
                              config=config,
                              espec_systematic=specdata[0].spec * 0.01)


if __name__ == '__main__':
    test_fits()

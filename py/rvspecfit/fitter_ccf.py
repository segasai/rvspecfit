import sys
import os
import pickle
import time
import numpy as np
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt
from rvspecfit import make_ccf


class CCFCache:
    """ Singleton caching CCF information """
    ccfs = {}


def get_ccf_info(spec_setup, config):
    """
    Returns the CCF info from the pickled file for a given spectroscopic spec_setup

    Parameters:
    -----------
    spec_setup: string
        The spectroscopic setup needed
    config: dict
        The dictionary with the config

    Returns:
    -------
    d: dict
        The dictionary with the CCF Information as saved by the make_ccf code

    """
    if spec_setup not in CCFCache.ccfs:
        CCFCache.ccfs[spec_setup] = pickle.load(
            open(config['template_lib']['ccffile'] % spec_setup, 'rb'))
    return CCFCache.ccfs[spec_setup]


def fit(specdata, config):
    """
    Process the data by doing cross-correlation with templates

    Parameters:
    -----------
    specdata: list of SpecData objects
        The list of data that needs to be fitted from differetn spectral
        setups.
    config: dict
        The configuration dictionary

    Returns:
    results: dict
        The dictionary with results such as best template parameters, best velocity
        best vsini.
    """
    # configuration parameters

    maxvel = 1000
    # only search for CCF peaks from -maxvel to maxvel
    nvelgrid = 2000
    # number of points on the ccf in the specified velocity range

    loglam = {}
    velstep = {}
    spec_fftconj = {}
    vels = {}
    off = {}
    subind = {}
    ccfs = {}
    proc_specs = {}

    for cursd in specdata:
        spec_setup = cursd.name
        lam = cursd.lam
        spec = cursd.spec
        espec = cursd.espec
        ccfs[spec_setup] = get_ccf_info(spec_setup, config)
        ccfconf = ccfs[spec_setup]['ccfconf']
        logl0 = ccfconf.logl0
        logl1 = ccfconf.logl1
        npoints = ccfconf.npoints
        proc_spec = make_ccf.preprocess_data(
            lam, spec, espec, badmask=cursd.badmask, ccfconf=ccfconf)
        proc_spec /= proc_spec.std()
        proc_specs[spec_setup] = proc_spec
        spec_fft = np.fft.fft(proc_spec)
        spec_fftconj[spec_setup] = spec_fft.conj()
        velstep[spec_setup] = (np.exp((logl1 - logl0) / npoints) - 1) * 3e5
        l = len(spec_fft)
        off[spec_setup] = l // 2
        vels[spec_setup] = ((np.arange(l) + off[spec_setup]) %
                            l - off[spec_setup]) * velstep[spec_setup]
        vels[spec_setup] = -np.roll(vels[spec_setup], off[spec_setup])
        subind[spec_setup] = np.abs(vels[spec_setup]) < maxvel

    maxv = -1e20
    best_id = -90
    best_v = -777

    nfft = ccfs[spec_setup]['ffts'].shape[0]
    vel_grid = np.linspace(-maxvel, maxvel, nvelgrid)
    best_ccf = vel_grid * 0
    for id in range(nfft):
        curccf = {}
        for spec_setup in ccfs.keys():
            curf = ccfs[spec_setup]['ffts'][id, :]
            ccf = np.fft.ifft(spec_fftconj[spec_setup] * curf).real
            ccf = np.roll(ccf, off[spec_setup])
            curccf[spec_setup] = ccf[subind[spec_setup]]
            curccf[spec_setup] = scipy.interpolate.UnivariateSpline(
                vels[spec_setup][subind[spec_setup]][::-1], curccf[spec_setup][::-1], s=0)(vel_grid)
            # plot(vel_grid, curccf[spec_setup],xr=[-1000,1000])#np.roll(np.fft.ifft(curf*curf.conj()),off[spec_setup]))
            #plt.draw(); plt.pause(0.1)

        allccf = np.array([curccf[_] for _ in ccfs.keys()]).prod(axis=0)
        if allccf.max() > maxv:
            maxv = allccf.max()
            best_id = id
            best_v = vel_grid[np.argmax(allccf)]
            best_model = {}
            for spec_setup in ccfs.keys():
                best_model[spec_setup] = np.roll(
                    ccfs[spec_setup]['models'][id], int(best_v / velstep[spec_setup]))
            best_ccf = allccf
    try:
        assert(best_id >= 0)
    except:
        raise Exception('Cross-correlation step failed')

    best_par = ccfs[list(ccfs.keys())[0]]['params'][best_id]
    best_par = dict(zip(ccfs[spec_setup]['parnames'], best_par))

    best_vsini = ccfs[list(ccfs.keys())[0]]['vsinis'][best_id]

    result = {}
    result['best_par'] = best_par
    result['best_vel'] = best_v
    result['best_ccf'] = best_ccf
    result['best_vsini'] = best_vsini
    result['best_model'] = best_model
    result['proc_spec'] = proc_specs
    return result

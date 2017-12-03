import sys
import os
import pickle
import time
import numpy as np
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt
import make_ccf

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
    config: 
    """
    if spec_setup not in CCFCache.ccfs:
        CCFCache.ccfs[spec_setup] = pickle.load(
            open(config['template_lib']['ccffile'] % spec_setup, 'rb'))
    return CCFCache.ccfs[spec_setup]


def doitquick(specdata, config):
    loglam = {}
    stor = {}
    velstep = {}
    spec_fftconj = {}
    vels = {}
    off = {}
    subind = {}
    maxvel = 1000
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
    bestid = -90
    bestv = -777
    maxvel = 1000
    nvelgrid = 2000
    nfft = ccfs[spec_setup]['ffts'].shape[0]
    vel_grid = np.linspace(-maxvel, maxvel, nvelgrid)
    bestccf = vel_grid * 0
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
            bestid = id
            bestv = vel_grid[np.argmax(allccf)]
            bestmodel = {}
            for spec_setup in ccfs.keys():
                bestmodel[spec_setup] = np.roll(
                    ccfs[spec_setup]['models'][id], int(bestv / velstep[spec_setup]))
            bestccf = allccf
    bestpar = ccfs[list(ccfs.keys())[0]]['params'][bestid]
    bestpar = dict(zip(ccfs[spec_setup]['parnames'], bestpar))
    bestvsini = ccfs[list(ccfs.keys())[0]]['vsinis'][bestid]

    result = {}
    result['bestpar'] = bestpar
    result['bestvel'] = bestv
    result['bestccf'] = bestccf
    result['bestvsini'] = bestvsini
    result['bestmodel'] = bestmodel
    result['proc_spec'] = proc_specs
    return result

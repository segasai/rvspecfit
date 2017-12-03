import numpy as np
import scipy.optimize
import numpy
import sys
import matplotlib.pyplot as plt
import scipy.interpolate
import os
import time
import make_ccf
import pickle


class ccfCache:
    ccfs = {}

def getCcf(setup, config):
    if setup not in ccfCache.ccfs:
        ccfCache.ccfs[setup] = pickle.load(open(config['template_lib']['ccffile'] % setup,'rb'))
    return ccfCache.ccfs[setup]


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
    for cursd in specdata:
        spec_setup = cursd.name
        lam = cursd.lam
        spec = cursd.spec
        espec = cursd.espec
        ccfs[spec_setup]= getCcf(spec_setup, config)
        ccfconf = ccfs[spec_setup]['ccfconf']
        loglam = np.linspace(
            numpy.log(lam[0]) + .01, numpy.log(lam[-1]) - 0.01, len(lam))
        logl0 = loglam[0]
        logl1 = loglam[-1]
        npoints = len(loglam)
        proc_spec = make_ccf.preprocess_data(
            lam, spec, espec, badmask=cursd.badmask, ccfconf=ccfconf)
        proc_spec /= proc_spec.std()
        spec_fft = np.fft.fft(proc_spec)
        spec_fftconj[spec_setup] = spec_fft.conj()
        velstep[spec_setup] = (np.exp(loglam[1] - loglam[0]) - 1) * 3e5
        l = len(spec_fft)
        off[spec_setup] = l // 2
        vels[spec_setup] = ((np.arange(l) + off[spec_setup]) % l - off[spec_setup]) * velstep[spec_setup]
        vels[spec_setup] = -np.roll(vels[spec_setup], off[spec_setup])
        subind[spec_setup] = np.abs(vels[spec_setup]) < maxvel

    maxv = -1e20
    bestid = -90
    bestv = -777
    nfft = ccfs[spec_setup]['ffts'].shape[0]
    vel_grid = np.linspace(-700, 700, 500)
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
        allccf = np.array([curccf[_] for _ in ccfs.keys()]).prod(axis=0)
        if allccf.max() > maxv:
            maxv = allccf.max()
            bestid = id
            bestv = vel_grid[numpy.argmax(allccf)]
            bestmodel = {}
            for spec_setup in ccfs.keys():
                bestmodel[spec_setup] = np.roll(
                    ccfs[spec_setup]['models'][id], int(bestv / velstep[spec_setup]))
            bestccf = allccf
    bestpar = ccfs[list(ccfs.keys())[0]]['params'][bestid]
    result = {}
    result['bestpar'] = bestpar
    result['bestvel'] = bestv
    result['bestccf'] = bestccf
    return result

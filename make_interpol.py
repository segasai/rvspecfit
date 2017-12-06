import multiprocessing as mp
import os
import subprocess
import sys
import argparse
import pickle
import matplotlib.pyplot as plt
import scipy.constants
import scipy.optimize
import numpy as np
import atpy
import read_grid
import utils

git_rev = utils.get_revision()


def get_line_continuum(lam, spec):
    """
    Determine the extremely simple linear in log continuum to
    remove away continuum trends in templates

    Parameters:
    -----------
    lam: numpy array
        Wavelength vector
    spec: numpy array
        spectrum

    Returns:
    --------
    cont: numpy array
        Continuum model

    """
    npix = len(lam)
    npix2 = npix // 2
    lam1, lam2 = [np.median(_) for _ in [lam[:npix2], lam[npix2:]]]
    sp1, sp2 = [np.median(_) for _ in [spec[:npix2], spec[npix2:]]]
    cont = np.exp(scipy.interpolate.UnivariateSpline([lam1, lam2],
                                                     np.log(np.r_[sp1, sp2]),
                                                     s=0, k=1, ext=0)(lam))
    return cont


class si:
    mat = None
    lamgrid = None


def extract_spectrum(logg, teff, feh, alpha, dbfile, prefix, wavefile):
    """
    Exctract a spectrum of a given parameters then apply the resolution smearing
    and divide by the continuum

    Parameters:
    -----------

    logg: real
        Surface gravity
    teff: real
        Effective Temperature
    feh: real
        Metallicity
    alpha: real
        Alpha/Fe
    dbfile: string
        Path to the sqlite database
    prefix: string
        Prefix to the data files
    wavefile: string
        Path to the file with wavelengths
    """

    lam, spec = read_grid.get_spec(
        logg, teff, feh, alpha, dbfile=dbfile, prefix=prefix, wavefile=wavefile)
    spec = read_grid.apply_rebinner(si.mat, spec)
    spec1 = spec / get_line_continuum(si.lamgrid, spec)
    spec1 = np.log(spec1)  # log the spectrum
    if not np.isfinite(spec1).all():
        raise Exception('nans %s' % str((teff, logg, feh, alpha)))
    spec1 = spec1.astype(np.float32)
    return spec1


def process_all(setupInfo, postf='', dbfile='/tmp/files.db', oprefix='psavs/',
                prefix=None, wavefile=None, air=False):
    nthreads = 8
    tab = atpy.Table('sqlite', dbfile)
    ids = (tab.id).astype(int)
    vec = np.array((tab.teff, tab.logg, tab.met, tab.alpha))
    parnames = ('teff', 'logg', 'feh', 'alpha')
    i = 0

    templ_lam, spec = read_grid.get_spec(4.5, 12000, 0, 0, dbfile=dbfile,
                                         prefix=prefix, wavefile=wavefile)

    HR, lamleft, lamright, resol, step, log = setupInfo

    deltav = 1000.  # extra padding
    fac1 = (1 + deltav / (scipy.constants.speed_of_light / 1e3))
    if not log:
        lamgrid = np.arange(lamleft / fac1, (lamright + step) * fac1, step)
    else:
        lamgrid = np.exp(np.arange(np.log(lamleft / fac1),
                                   np.log(lamright * fac1), np.log(1 + step / lamleft)))

    mat = read_grid.make_rebinner(templ_lam, lamgrid, resol, toair=air)

    specs = []
    si.mat = mat
    si.lamgrid = lamgrid
    pool = mp.Pool(nthreads)
    for curteff, curlogg, curfeh, curalpha in vec.T:
        curid = ids[i]
        i += 1
        print(i)
        specs.append(pool.apply_async(
            extract_spectrum, (curlogg, curteff, curfeh, curalpha,
                               dbfile, prefix, wavefile)))
    lam = lamgrid
    for i in range(len(specs)):
        specs[i] = specs[i].get()

    pool.close()
    pool.join()
    specs = np.array(specs)
    with open('%s/specs_%s%s.pkl' % (oprefix, HR, postf), 'wb') as fp:
        pickle.dump(dict(specs=specs, vec=vec, lam=lam,
                         parnames=parnames, git_rev=git_rev), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--setup', type=str)
    parser.add_argument('--lambda0', type=float)
    parser.add_argument('--lambda1', type=float)
    parser.add_argument('--resol', type=float)
    parser.add_argument('--step', type=float)
    parser.add_argument('--log', action='store_true', default=True)
    parser.add_argument('--templdb', type=str, default='files.db')
    parser.add_argument('--templprefix', type=str)
    parser.add_argument('--air', action='store_true', default=False)
    parser.add_argument('--oprefix', type=str, default='templ_data/')
    parser.add_argument('--wavefile', type=str)
    args = parser.parse_args()

    process_all((args.setup, args.lambda0, args.lambda1,
                 args.resol, args.step, args.log), dbfile=args.templdb, oprefix=args.oprefix,
                prefix=args.templprefix,
                wavefile=args.wavefile, air=args.air)

from __future__ import print_function
import multiprocessing as mp
import os
import subprocess
import sys
import argparse
import pickle
import scipy.constants
import scipy.optimize
import numpy as np
import sqlite3
from rvspecfit import read_grid
from rvspecfit import utils
from rvspecfit import _version
git_rev = _version.VERSION

SPEC_PKL_NAME = 'specs_%s.pkl'


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
    cont = np.exp(
        scipy.interpolate.UnivariateSpline(
            [lam1, lam2], np.log(np.r_[sp1, sp2]), s=0, k=1, ext=0)(lam))
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
        logg,
        teff,
        feh,
        alpha,
        dbfile=dbfile,
        prefix=prefix,
        wavefile=wavefile)
    spec = read_grid.apply_rebinner(si.mat, spec)
    spec1 = spec / get_line_continuum(si.lamgrid, spec)
    spec1 = np.log(spec1)  # log the spectrum
    if not np.isfinite(spec1).all():
        raise Exception(
            'The spectrum is not finite (has nans or infs) at parameter values: %s'
            % str((teff, logg, feh, alpha)))
    spec1 = spec1.astype(np.float32)
    return spec1


def process_all(setupInfo,
                postf='',
                dbfile='/tmp/files.db',
                oprefix='psavs/',
                prefix=None,
                wavefile=None,
                air=False,
                resolution0=None,
                fixed_fwhm=False):
    nthreads = 8
    conn = sqlite3.connect(dbfile)
    cur = conn.execute('select id, teff, logg, met, alpha from files')
    tab = np.rec.fromrecords(cur.fetchall())
    tab_id, tab_teff, tab_logg, tab_met, tab_alpha = tab.f0, tab.f1, tab.f2, tab.f3, tab.f4
    ids = (tab_id).astype(int)
    vec = np.array((tab_teff, tab_logg, tab_met, tab_alpha))
    parnames = ('teff', 'logg', 'feh', 'alpha')
    i = 0

    templ_lam, spec = read_grid.get_spec(
        4.5, 12000, 0, 0, dbfile=dbfile, prefix=prefix, wavefile=wavefile)
    mapper = read_grid.ParamMapper()
    HR, lamleft, lamright, resol, step, log = setupInfo

    deltav = 1000.  # extra padding
    fac1 = (1 + deltav / (scipy.constants.speed_of_light / 1e3))
    if not log:
        lamgrid = np.arange(lamleft / fac1, (lamright + step) * fac1, step)
    else:
        lamgrid = np.exp(
            np.arange(
                np.log(lamleft / fac1), np.log(lamright * fac1),
                np.log(1 + step / lamleft)))

    mat = read_grid.make_rebinner(
        templ_lam,
        lamgrid,
        resol,
        toair=air,
        resolution0=resolution0,
        fixed_fwhm=fixed_fwhm)

    specs = []
    si.mat = mat
    si.lamgrid = lamgrid
    pool = mp.Pool(nthreads)
    for curteff, curlogg, curfeh, curalpha in vec.T:
        curid = ids[i]
        i += 1
        print(i)
        specs.append(
            pool.apply_async(extract_spectrum,
                             (curlogg, curteff, curfeh, curalpha, dbfile,
                              prefix, wavefile)))
    lam = lamgrid
    for i in range(len(specs)):
        specs[i] = specs[i].get()

    pool.close()
    pool.join()
    specs = np.array(specs)
    with open(('%s/' + SPEC_PKL_NAME) % (oprefix, HR), 'wb') as fp:
        pickle.dump(
            dict(
                specs=specs,
                vec=vec,
                lam=lam,
                parnames=parnames,
                git_rev=git_rev,
                mapper=mapper), fp)


def main(args):
    parser = argparse.ArgumentParser(
        description=
        'Create interpolated and convolved spectra from the input grid.')
    parser.add_argument(
        '--setup', type=str, help='Name of the spectral configuration')
    parser.add_argument('--lambda0', type=float, help='Start wavelength')
    parser.add_argument('--lambda1', type=float, help='End wavelength')
    parser.add_argument('--resol', type=float, help='Spectral resolution R')
    parser.add_argument('--step', type=float, help='Pixel size in angstrom')
    parser.add_argument(
        '--log',
        action='store_true',
        default=True,
        help='Generate spectra in log-waelength scale')
    parser.add_argument(
        '--templdb',
        type=str,
        default='files.db',
        help='The path to the SQLiteDB with the info about the templates')
    parser.add_argument(
        '--templprefix', type=str, help='The path to the templates')
    parser.add_argument(
        '--air',
        action='store_true',
        default=False,
        help='Generate spectra in the air (rather than vacuum) frame')
    parser.add_argument(
        '--oprefix',
        type=str,
        default='templ_data/',
        help='The path where the converted templates will be created')
    parser.add_argument(
        '--wavefile',
        type=str,
        help=
        'The path to the fits file with the wavelength grid of the templates')
    parser.add_argument(
        '--resolution0',
        type=float,
        default=100000,
        help='The resolution of the input grid')
    parser.add_argument(
        '--fixed_fwhm',
        action='store_true',
        default=False,
        help=
        'Use to make the fwhm of the LSF to be constant rather then R=lambda/dlambda'
    )

    args = parser.parse_args(args)

    process_all(
        (args.setup, args.lambda0, args.lambda1, args.resol, args.step,
         args.log),
        dbfile=args.templdb,
        oprefix=args.oprefix,
        prefix=args.templprefix,
        wavefile=args.wavefile,
        air=args.air,
        resolution0=args.resolution0,
        fixed_fwhm=args.fixed_fwhm)


if __name__ == '__main__':
    main(sys.argv[1:])

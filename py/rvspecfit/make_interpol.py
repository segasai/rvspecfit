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
import rvspecfit
git_rev = rvspecfit.__version__

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
        scipy.interpolate.UnivariateSpline([lam1, lam2],
                                           np.log(np.r_[sp1, sp2]),
                                           s=0,
                                           k=1,
                                           ext=0)(lam))
    return cont


class si:
    # cached sparse convolution matrix and the output wavelength grid
    mat = None
    lamgrid = None


def extract_spectrum(logg,
                     teff,
                     feh,
                     alpha,
                     dbfile,
                     prefix,
                     wavefile,
                     normalize=True):
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
    normalize: boolean
        Normalize the spectrum by a linear continuum
    """

    lam, spec0 = read_grid.get_spec(logg,
                                    teff,
                                    feh,
                                    alpha,
                                    dbfile=dbfile,
                                    prefix=prefix,
                                    wavefile=wavefile)
    # Here I assume that the input spectrum is in erg/wavelength
    # I will now convert into number of photons before convolving
    # with the resolution vector
    spec0_phot = spec0 * lam
    spec1_phot = read_grid.apply_rebinner(si.mat, spec0_phot)
    spec1 = spec1_phot / si.lamgrid

    normnum = np.median(spec1)
    spec2 = spec1 / normnum
    if normalize:
        spec2 = spec2 / get_line_continuum(si.lamgrid, spec2)

    spec2 = np.log(spec2)  # log the spectrum
    if not np.isfinite(spec2).all():
        raise Exception(
            'The spectrum is not finite (has nans or infs) at parameter values: %s'
            % str((teff, logg, feh, alpha)))
    spec2 = spec2.astype(np.float32)
    return spec2, np.log(normnum)


class Resolution:
    # resolution object returning an array
    def __init__(self, resol=None, resol_func=None):
        self.resol = resol
        self.resol_func = resol_func
        assert (self.resol is not None or self.resol_func is not None)

    def __call__(self, x):
        if self.resol is None:
            return eval(self.resol_func, dict(x=x, np=np))
        else:
            return self.resol


def process_all(setupInfo,
                postf='',
                dbfile='/tmp/files.db',
                oprefix='psavs/',
                prefix=None,
                wavefile=None,
                air=False,
                resolution0=None,
                normalize=True,
                revision=''):
    nthreads = 8
    if not os.path.exists(dbfile):
        raise Exception('The template database file %s does not exist' %
                        dbfile)
    conn = sqlite3.connect(dbfile)
    cur = conn.execute('select id, teff, logg, met, alpha from files')
    tab = np.rec.fromrecords(cur.fetchall())
    tab_id, tab_teff, tab_logg, tab_met, tab_alpha = tab.f0, tab.f1, tab.f2, tab.f3, tab.f4
    ids = (tab_id).astype(int)
    nspec = len(ids)
    vec = np.array((tab_teff, tab_logg, tab_met, tab_alpha))
    parnames = ('teff', 'logg', 'feh', 'alpha')
    i = 0

    templ_lam, spec = read_grid.get_spec(4.5,
                                         12000,
                                         0,
                                         0,
                                         dbfile=dbfile,
                                         prefix=prefix,
                                         wavefile=wavefile)
    mapper = read_grid.ParamMapper()
    HR, lamleft, lamright, resol_function, step, log = setupInfo

    deltav = 1000.  # extra padding
    fac1 = (1 + deltav / (scipy.constants.speed_of_light / 1e3))
    if not log:
        lamgrid = np.arange(lamleft / fac1, (lamright + step) * fac1, step)
    else:
        lamgrid = np.exp(
            np.arange(np.log(lamleft / fac1), np.log(lamright * fac1),
                      np.log(1 + step / lamleft)))

    mat = read_grid.make_rebinner(templ_lam,
                                  lamgrid,
                                  resol_function,
                                  toair=air,
                                  resolution0=resolution0)

    specs = []
    lognorms = np.zeros(nspec)
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
                              prefix, wavefile, normalize)))
    lam = lamgrid
    for i in range(len(specs)):
        specs[i], lognorms[i] = specs[i].get()

    pool.close()
    pool.join()
    specs = np.array(specs)
    lognorms = np.array(lognorms)

    if os.path.isdir(oprefix):
        pass
    else:
        try:
            os.mkdir(oprefix)
        except:
            raise Exception('Failed to create output directory: %s' %
                            (oprefix, ))
    with open(('%s/' + SPEC_PKL_NAME) % (oprefix, HR), 'wb') as fp:
        pickle.dump(
            dict(specs=specs,
                 vec=vec,
                 lam=lam,
                 parnames=parnames,
                 git_rev=git_rev,
                 mapper=mapper,
                 revision=revision,
                 lognorms=lognorms), fp)


def add_bool_arg(parser, name, default=False, help=None):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true', help=help)
    group.add_argument('--no-' + name,
                       dest=name,
                       action='store_false',
                       help='Invert the ' + name + ' option')
    parser.set_defaults(**{name: default})
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/19233287


def main(args):
    parser = argparse.ArgumentParser(
        description=
        'Create interpolated and convolved spectra from the input grid.')
    parser.add_argument('--setup',
                        type=str,
                        help='Name of the spectral configuration',
                        required=True)
    parser.add_argument('--lambda0',
                        type=float,
                        help='Start wavelength of the new grid',
                        required=True)
    parser.add_argument('--lambda1',
                        type=float,
                        help='End wavelength of the new grid',
                        required=True)
    parser.add_argument('--resol',
                        type=float,
                        help='Spectral resolution of the new grid')
    parser.add_argument('--revision',
                        type=str,
                        help='The revision of the templates',
                        default='',
                        required=False)

    parser.add_argument(
        '--resol_func',
        type=str,
        help=
        'Spectral resolution function of the new grid. It is a string that should be a function of wavelength in angstrom, i.e. 1000+2*x ',
    )
    parser.add_argument('--step',
                        type=float,
                        help='Pixel size in angstrom of the new grid',
                        required=True)
    add_bool_arg(parser,
                 'log',
                 default=True,
                 help='Generate the spectra in log-wavelength scale')
    add_bool_arg(parser,
                 'normalize',
                 default=True,
                 help='Normalize the spectra')

    parser.add_argument(
        '--templdb',
        type=str,
        default='files.db',
        help='The path to the SQLiteDB with the info about the templates')
    parser.add_argument('--templprefix',
                        type=str,
                        help='The path to the templates',
                        required=True)
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
    parser.add_argument('--resolution0',
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

    if not (args.resol is not None or args.resol_func is not None):
        parser.error('Either --resol or --resol_func is required')
    if (args.resol is not None and args.resol_func is not None):
        parser.error('Either --resol or --resol_func is required, not both')
    if (args.resol_func is not None and args.fixed_fwhm):
        parser.error('Either --resol_func is incompatible with --fixed_fwhm')

    if args.resol is not None:
        if args.fixed_fwhm:
            lam_mid = (args.lambda0 + args.lambda1) * .5
            resol_func = Resolution(resol_func='x/%f*%f' %
                                    (lam_mid, args.resol))
        else:
            resol_func = Resolution(resol=args.resol)
    else:
        resol_func = Resolution(resol_func=args.resol_func)

    process_all((args.setup, args.lambda0, args.lambda1, resol_func, args.step,
                 args.log),
                dbfile=args.templdb,
                oprefix=args.oprefix,
                prefix=args.templprefix,
                wavefile=args.wavefile,
                air=args.air,
                resolution0=args.resolution0,
                normalize=args.normalize,
                revision=args.revision)


if __name__ == '__main__':
    main(sys.argv[1:])

from __future__ import print_function
import multiprocessing as mp
import os
import sys
import argparse
import logging
import importlib
import scipy.constants
import scipy.optimize
import numpy as np
import sqlite3
from rvspecfit import read_grid
from rvspecfit import serializer
import rvspecfit

git_rev = rvspecfit.__version__

SPECS_H5_NAME = 'specs_%s.h5'


class FakePoolResult:

    def __init__(self, x):
        self.x = x

    def get(self):
        return self.x


class FakePool:

    def __init__(self):
        pass

    def apply_async(self, func, args, kwargs={}):
        return FakePoolResult(func(*args, **kwargs))

    def close(self):
        pass

    def join(self):
        pass


def get_line_continuum(lam, spec):
    """
    Determine the extremely simple linear in log continuum to
    remove away continuum trends in templates

    Parameters
    -----------
    lam: numpy array
        Wavelength vector
    spec: numpy array
        spectrum

    Returns
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


def initialize_matrix_cache(mat, lamgrid):
    si.mat = mat
    si.lamgrid = lamgrid


def get_mapper(mapper_module, mapper_class_name, mapper_args):
    mod = importlib.import_module(mapper_module)
    return getattr(mod, mapper_class_name)(*mapper_args)


def extract_spectrum(param,
                     dbfile,
                     prefix,
                     wavefile,
                     normalize=True,
                     log_spec=True):
    """
    Extract a spectrum of a given parameters then apply the resolution
    smearing and divide by the continuum

    Parameters
    -----------
    param: dict
        The dictionary of key value pairs of parameters
    dbfile: string
        Path to the sqlite database
    prefix: string
        Prefix to the data files
    wavefile: string
        Path to the file with wavelengths
    normalize: boolean
        Normalize the spectrum by a linear continuum
    log_spec: boolean
        If True, take the logarithm of the spectrum

    Returns
    -------
    spec: numpy array
        The processed spectrum (logarithmic if log_spec=True)
    lognorm: float
        The logarithm of the normalization factor
    """

    lam, spec0 = read_grid.get_spec(param,
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
    if log_spec:
        spec2 = np.log(spec2)  # log the spectrum
    if not np.isfinite(spec2).all():
        raise RuntimeError('The spectrum is not finite (has nans or infs) at '
                           'parameter values: %s' % str(param))
    spec2 = spec2.astype(np.float32)
    return spec2, np.log(normnum)


class Resolution:
    # resolution object returning an array
    def __init__(self, resol=None, resol_func=None):
        self.resol = resol
        self.resol_func = resol_func
        assert (self.resol is not None or self.resol_func is not None)

    def __call__(self, x):
        """Evaluate resolution at fixed wavelength

        """
        if self.resol is None:
            return eval(self.resol_func, dict(x=x, np=np))
        else:
            return self.resol


def _fetch_all_parameters(dbfile, parnames):
    """
    Read the vector of template parameters corresponding to parnames
    """
    if not os.path.exists(dbfile):
        raise RuntimeError('The template database file %s does not exist' %
                           dbfile)

    parname_str = ','.join(list(parnames))
    with sqlite3.connect(dbfile) as conn:
        table_exists = conn.execute('''SELECT count(*) FROM
        sqlite_schema WHERE type='table' AND name='grid_parameters' '''
                                    ).fetchall()[0][0] == 1
        if table_exists:
            nparam = conn.execute(
                'select count(*) from grid_parameters').fetchall()[0][0]
            if nparam != len(parnames):
                raise RuntimeError(
                    'You did not specify the correct number of grid '
                    f'parameters (the database says there are {nparam})')
        else:
            logging.warning(
                'You are using an older format database it may be wise '
                'to upgrade by rerunning read_grid')
        cur = conn.execute(f'''select id, {parname_str} from files
        where not bad  order by {parname_str}''')
        tab = np.rec.fromrecords(cur.fetchall())
    vec = np.array([tab['f%d' % _] for _ in range(1, len(parnames) + 1)])
    return vec


def process_all(setupInfo,
                parnames=('teff', 'logg', 'feh', 'alpha'),
                dbfile='/tmp/files.db',
                oprefix='psavs/',
                prefix=None,
                wavefile=None,
                air=False,
                resolution0=None,
                normalize=True,
                revision='',
                nthreads=8,
                log_parameters=None):
    """
    Process the whole library of spectra and prepare the pickle file
    with arrays of convolved spectra, wavelength arrays, transformed
    parameters

    Parameters
    -----------
    setupInfo: string
        The name of spectral configuration
    parnames: list of strings
        The parameter names of spectra
    log_parameters: integer positions of parameters that needs
        to be log10() for interpolation. I.e. if the first parameter si teff
        and we want to perform interpolation in log(teff) space
        this needs to be [0]
    air: boolean
        Transform from vacuum to air

    """
    vec = _fetch_all_parameters(dbfile, parnames)
    nspec = vec.shape[1]
    log_spec = True
    i = 0

    par0 = dict(zip(parnames, vec.T[0]))
    templ_lam, spec = read_grid.get_spec(par0,
                                         dbfile=dbfile,
                                         prefix=prefix,
                                         wavefile=wavefile)
    mapper_module = 'rvspecfit.read_grid'
    mapper_class = 'LogParamMapper'
    mapper_args = (log_parameters, )
    HR, lamleft, lamright, resol_function, step, log = setupInfo
    if templ_lam.min() > lamleft or templ_lam.max() < lamright:
        raise RuntimeError(f'''Cannot generate the spectra as the wavelength
        range in the library does not cover the requested wavelengths
        {lamleft} {lamright} {templ_lam.min()} {templ_lam.max()}
        ''')

    deltav = 1000.  # extra padding
    fac1 = (1 + deltav / (scipy.constants.speed_of_light / 1e3))
    if not log:
        lamgrid = np.arange(lamleft / fac1, (lamright + step) * fac1, step)
    else:
        logstep = np.log(1 + step / (0.5 * (lamleft + lamright)))
        # logstep is such that  it correspond to linear step step
        # for the middle of the wavelength range
        lamgrid = np.exp(
            np.arange(np.log(lamleft / fac1), np.log(lamright * fac1),
                      logstep))
    if len(lamgrid) <= 1:
        raise RuntimeError(
            'Did you incorrectly specify wavelength range or step ? ')
    mat = read_grid.make_rebinner(templ_lam,
                                  lamgrid,
                                  resol_function,
                                  toair=air,
                                  resolution0=resolution0)

    specs = []
    lognorms = np.zeros(nspec)
    if nthreads > 1:
        multi_thread = True
        pool = mp.Pool(nthreads, initialize_matrix_cache, (mat, lamgrid))
    else:
        multi_thread = False
        pool = FakePool()
        initialize_matrix_cache(mat, lamgrid)
    for curvec in vec.T:
        i += 1
        param = dict(zip(parnames, curvec))
        if not multi_thread:
            if i % max(1, nspec // 100) == 0:
                print('%d/%d' % (i, nspec))
        specs.append(
            pool.apply_async(
                extract_spectrum,
                (param, dbfile, prefix, wavefile, normalize, log_spec)))
    lam = lamgrid
    for i in range(len(specs)):
        if multi_thread:
            if i % max(1, nspec // 100) == 0:
                print('%d/%d' % (i, nspec))
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
        except OSError:
            raise RuntimeError('Failed to create output directory: %s' %
                               (oprefix, ))
    curfname = ('%s/' + SPECS_H5_NAME) % (oprefix, HR)
    DD = dict(specs=specs,
              vec=vec,
              lam=lam,
              parnames=parnames,
              git_rev=git_rev,
              mapper_module=mapper_module,
              mapper_class_name=mapper_class,
              mapper_args=mapper_args,
              revision=revision,
              lognorms=lognorms,
              log_step=log,
              log_spec=log_spec)
    serializer.save_dict_to_hdf5(curfname, DD)


def add_bool_arg(parser, name, default=False, help=None):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true', help=help)
    group.add_argument('--no-' + name,
                       dest=name,
                       action='store_false',
                       help='Invert the ' + name + ' option')
    parser.set_defaults(**{name: default})
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/19233287


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description='Create interpolated and convolved spectra from the '
        'input grid.')
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
                        help='Constant spectral resolution of the new grid')
    parser.add_argument('--revision',
                        type=str,
                        help='The revision of the templates',
                        default='',
                        required=False)
    parser.add_argument(
        '--parameter_names',
        type=str,
        default='teff,logg,feh,alpha',
        help='comma separated list of parameters to make the interpolator',
        required=False)

    parser.add_argument(
        '--log_parameters',
        type=str,
        default='0',
        help='Which parameters we are taking the log() of when interpolating',
        required=False)

    parser.add_argument(
        '--resol_func',
        type=str,
        help=(
            'Spectral resolution function of the new grid. '
            'It is a string that '
            'should be a function of wavelength in angstrom, i.e. 1000+2*x. ' +
            'This option is incompatible with --resol'),
    )
    parser.add_argument(
        '--step',
        type=float,
        help=('Pixel size in angstrom of the templates in the grid ' +
              'If log-spacing is requested (default) the pixel size would ' +
              'correspond to the pixel size for smallest wavelengths'),
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
        required=True,
        help='The path to the fits file with the wavelength grid '
        'of templates')
    parser.add_argument('--resolution0',
                        type=float,
                        default=100000,
                        help='The resolution of the input grid')
    parser.add_argument('--nthreads',
                        type=int,
                        default=8,
                        help='The number of threads used')
    parser.add_argument(
        '--fixed_fwhm',
        action='store_true',
        default=False,
        help=('Use this option to make the FWHM of the LSF to be constant '
              'rather then resolution R=lambda/dlambda'))

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

    log_parameters = [int(_) for _ in args.log_parameters.split(',')]

    parnames = args.parameter_names.split(',')

    process_all((args.setup, args.lambda0, args.lambda1, resol_func, args.step,
                 args.log),
                parnames=parnames,
                log_parameters=log_parameters,
                dbfile=args.templdb,
                oprefix=args.oprefix,
                prefix=args.templprefix,
                wavefile=args.wavefile,
                air=args.air,
                resolution0=args.resolution0,
                normalize=args.normalize,
                revision=args.revision,
                nthreads=args.nthreads)


if __name__ == '__main__':
    main()

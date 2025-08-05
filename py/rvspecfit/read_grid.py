import argparse
import itertools
import functools
import warnings
import glob
import sys
import os
import sqlite3
import astropy.io.fits as pyfits
import scipy.stats
import scipy.sparse
import numpy as np


def gau_integrator(A, B, x1, x2, l1, l2, s):
    """ This computes the integral of (Ax+B)/sqrt(2pi)/s*exp(-1/2*(x-y)^2/s^2
    for x=x1..x2 y=l1..l2

    Here is the Mathematica code
    FortranForm[
    Simplify[Integrate[(A*x + B)/Sqrt[2*Pi]/s*
    Exp[-1/2*(x - y)^2/s^2], {x, x1, x2}, {y, l1, l2}]]]

    Parameters
    A: ndarray
    B: ndarray
    x1: ndarray
    x2: ndarray
    l1: ndarray
    l2: ndarray
    s: ndarray

    Returns
    -------
    ret: float
        Integral of the function

    """
    E = np.exp(1)
    Sqrt = np.sqrt
    Erf = scipy.special.erf
    Pi = np.pi
    return (-((Sqrt(2 / Pi) * s *
               (2 * B + A * (l1 + x1))) * E**(-(l1 - x1)**2 / (2. * s**2))) +
            (Sqrt(2 / Pi) * s * (2 * B + A * (l2 + x1))) * E**(-(l2 - x1)**2 /
                                                               (2. * s**2)) +
            (Sqrt(2 / Pi) * s * (2 * B + A * (l1 + x2))) * E**(-(l1 - x2)**2 /
                                                               (2. * s**2)) -
            (Sqrt(2 / Pi) * s *
             (2 * B + A * (l2 + x2))) * E**(-(l2 - x2)**2 / (2. * s**2)) + x1 *
            (2 * B + A * x1) * Erf(
                (l1 - x1) / (Sqrt(2) * s)) - x1 * (2 * B + A * x1) * Erf(
                    (l2 - x1) / (Sqrt(2) * s)) + 2 * B * l1 * Erf(
                        (-l1 + x1) / (Sqrt(2) * s)) + A * l1**2 * Erf(
                            (-l1 + x1) / (Sqrt(2) * s)) + A * s**2 * Erf(
                                (-l1 + x1) / (Sqrt(2) * s)) - 2 * B * l2 * Erf(
                                    (-l2 + x1) /
                                    (Sqrt(2) * s)) - A * l2**2 * Erf(
                                        (-l2 + x1) /
                                        (Sqrt(2) * s)) - A * s**2 * Erf(
                                            (-l2 + x1) / (Sqrt(2) * s)) - x2 *
            (2 * B + A * x2) * Erf(
                (l1 - x2) / (Sqrt(2) * s)) + x2 * (2 * B + A * x2) * Erf(
                    (l2 - x2) / (Sqrt(2) * s)) - 2 * B * l1 * Erf(
                        (-l1 + x2) / (Sqrt(2) * s)) - A * l1**2 * Erf(
                            (-l1 + x2) / (Sqrt(2) * s)) - A * s**2 * Erf(
                                (-l1 + x2) / (Sqrt(2) * s)) + 2 * B * l2 * Erf(
                                    (-l2 + x2) /
                                    (Sqrt(2) * s)) + A * l2**2 * Erf(
                                        (-l2 + x2) /
                                        (Sqrt(2) * s)) + A * s**2 * Erf(
                                            (-l2 + x2) / (Sqrt(2) * s))) / 4.


def pix_integrator(x1, x2, l1, l2, s):
    """ Integrate the flux within the pixel given the LSF
    We assume that the flux before LSF convolution is given by linear
    interpolation from x1, x2.
    The l1,l2 are scalar edges of the pixel within which we want to compute
    the flux. s is the sigma of the LSF
    The function returns two values of weights for y1,y2 where y1,y2 are
    the values of the non-convolved spectra at x1,x2

    Parameters
    ----------
    x1: ndarray
    x2: ndarray
    l1: ndarray
    l2: ndarray
    s: ndarray

    Results
    -------
    ret: tuple of ndarray
        Two weights for the integral (c1,c2) i.e. if the fluxes at x1,x2 are
        f1, f2 the integral is c1*f1+c2*f2

    """
    # the reason for two integrations because we have a function
    # (f1*(-x+x2)+ f2*(x-x1))/(x2-x1)
    #
    offset = x1
    # I put in offset, as integration is ambivalent to offsets, but
    # gau_integrator has numerical issues when x1,x2 is large
    coeff1 = gau_integrator(-1. / (x2 - x1), (x2 - offset) / (x2 - x1),
                            x1 - offset, x2 - offset, l1 - offset, l2 - offset,
                            s)
    coeff2 = gau_integrator(1. / (x2 - x1), -(x1 - offset) / (x2 - x1),
                            x1 - offset, x2 - offset, l1 - offset, l2 - offset,
                            s)
    return coeff1, coeff2


class LogParamMapper:
    """
    Class used to map stellar atmospheric parameters into more suitable
    space used for interpolation
    """

    def __init__(self, log_ids):
        # Specify which parameter numbers to log() for
        # interpolation
        self.log_ids = log_ids

    def forward(self, vec):
        """
        Map atmospheric parameters into parameters used in the grid for
        interpolation. That includes logarithming the teff

        Parameters
        -----------
        vec: numpy array
            The vector of atmospheric parameters i.e. Teff, logg, feh, alpha

        Returns
        ----------
        ret: numpy array
            The vector of transformed parameters used in interpolation
        """
        vec1 = np.array(vec, dtype=np.float64)
        for i in self.log_ids:
            vec1[i] = np.log10(vec1[i])
        return vec1

    def inverse(self, vec):
        """
        Map transformed parameters used in the grid for interpolation back into
        the atmospheric parameters. That includes exponentiating the
        log10(teff)

        Parameters
        -----------
        vec: numpy array
            The vector of transformed atmospheric parameters
            log(Teff), logg, feh, alpha

        Returns
        ----------
        ret: numpy array
            The vector of original atmospheric parameters.
    """
        vec1 = np.array(vec, dtype=np.float64)
        for i in self.log_ids:
            vec1[i] = 10**(vec1[i])
        return vec1


def makedb(prefix='/PHOENIX-ACES-AGSS-COND-2011/',
           dbfile='files.db',
           keywords=None,
           mask=None,
           extra_params=None,
           name_metallicity=None,
           name_alpha=None):
    """ Create an sqlite database of the templates

    Parameters
    ----------
    prefix: str
        The path to PHOENIX
    dbfile: str
        The output file with the sqlite DB
    keywords: dict
        The dictionary with the map of teff,logg,feh,alpha to keyword names
        in the headers
    mask: string
        The string how to identify spectral files, i.e. '*/*fits'
    extra_params: dict or None
        The dictionary of variable name vs FITS name to read from spectral
        files
    """
    if os.path.exists(dbfile):
        print(f'Overwriting the template database file {dbfile}')
        os.unlink(dbfile)
    DB = sqlite3.connect(dbfile)
    id = 0
    extra_params_str = ''
    if extra_params is not None:
        extra_keys = []
        for k, v in extra_params.items():
            extra_params_str = extra_params_str + f'{k} real,'
            extra_keys.append(v)
    else:
        extra_params = {}

    # Create a table listing what grid parameters we have
    DB.execute('''CREATE TABLE grid_parameters(
    id int, name varchar, explanation varchar)''')
    for counter, k in enumerate(
            itertools.chain(keywords.keys(), extra_params.keys())):
        DB.execute('INSERT INTO grid_parameters (id, name) values (?, ?)',
                   (counter, k))
    DB.execute(f'''CREATE TABLE files (filename varchar,
        teff real,
        logg real,
        {name_metallicity} real,
        {name_alpha} real,
        {extra_params_str}
        id int,
        bad bool);''')

    fs = sorted(glob.glob(prefix + mask))

    if len(fs) == 0:
        raise Exception(
            "No FITS templates found in the directory specified (using mask %s"
            % mask)
    for f in fs:
        hdr = pyfits.getheader(f)
        curpar = {}
        for param, curkey in itertools.chain(keywords.items(),
                                             extra_params.items()):
            if curkey not in hdr:
                raise Exception(
                    f"Keyword for {param} {curkey} not found in {f}")
            curpar[param] = hdr[curkey]

        query = ('insert into files (filename, id, bad,' +
                 ','.join(curpar.keys()) + ') values (?, ?, ? ' +
                 len(curpar) * ',?' + ' )')
        DB.execute(query,
                   (f.replace(prefix, ''), id, False) + tuple(curpar.values()))
        id += 1
    DB.commit()


@functools.lru_cache(None)
def _get_dbconn(dbfile):
    conn = sqlite3.connect(dbfile)
    return conn


def get_spec(params, dbfile=None, prefix=None, wavefile=None):
    """ Returns individual spectra for a given spectral parameters

    Parameters
    ----------
    params: dict
        The dictionary of parameters
    dbfile: string
        The pathname to the database sqlite file of templates
    prefix: string
        The prefix path to templates
    wavefile: string
        The filename of fits file with the wavelength vector

    Returns
    -------
    lam: ndarray
        1-D array of wavelength
    spec: ndarray
        1-D array of spectrum
    Example
    -------
    >>> lam,spec=read_grid.get_spec(dict(logg=1,teff=5250,feh=-1,alpha=0.4))

    """

    # We don't look for equality we look around the value with the following
    # deltas

    query = '''select filename from files where '''
    for ii, (k, v) in enumerate(params.items()):
        pad = 0.01
        v1 = v - pad
        v2 = v + pad
        if ii > 0:
            query += ' and '
        query += (f' {k} between {v1} and {v2} ')

    conn = _get_dbconn(dbfile)
    cur = conn.cursor()
    cur.execute(query)
    fnames = cur.fetchall()
    if len(fnames) > 1:
        print('Warning: More than 1 file returned', file=sys.stderr)
    if len(fnames) == 0:
        raise Exception('No spectra found')

    dat = pyfits.getdata(prefix + '/' + fnames[0][0])
    dat = dat.astype(dat.dtype.newbyteorder('='))  # convert to native
    lams = pyfits.getdata(wavefile)
    lams = lams.astype(lams.dtype.newbyteorder('='))
    return lams, dat


def make_rebinner(lam00,
                  lam,
                  resolution_function,
                  resolution0=None,
                  toair=True):
    """
    Make the sparse matrix that convolves a given spectrum to
    a given resolution and new wavelength grid

    Parameters
    -----------
    lam00: array
        The input wavelength grid of the templates
    lam: array
        The desired wavelength grid of the output
    resolution_function: function
        The function that when called with the wavelength as an argument
        will return the resolution of the desired spectra (R=l/dl)
        I.e. it could be just lambda x: 5000
    toair: bool
        Convert the input spectra into the air frame
    resolution0: float
        The resolution of input templates

    Returns
    --------
    The sparse matrix to perform interpolation
    """
    if toair:
        lam0 = lam00 / (1.0 + 2.735182E-4 + 131.4182 / lam00**2 +
                        2.76249E8 / lam00**4)
    else:
        lam0 = lam00

    resolution_array = resolution_function(lam)
    resolution_array = resolution_array + lam * 0  # ensure it is an array
    assert (resolution_array.max() < resolution0)
    fwhms = lam / resolution_array
    fwhms0 = lam / resolution0
    fwhm_to_sig = 2 * np.sqrt(2 * np.log(2))
    sigs = (fwhms**2 - fwhms0**2)**.5 / fwhm_to_sig
    thresh = 5  # 5 sigma
    xs = []
    ys = []
    vals = []
    size_warning = False
    for i in range(len(lam)):
        # we iterate over the output pixels
        curlam = lam[i]
        if i > 0:
            leftstep = 0.5 * (lam[i] - lam[i - 1])
        else:
            leftstep = 0.5 * (lam[i + 1] - lam[i])
        if i < len(lam) - 1:
            rightstep = 0.5 * (lam[i + 1] - lam[i])
        else:
            rightstep = leftstep
        cursig = sigs[i]
        curl0 = curlam - thresh * cursig
        curl1 = curlam + thresh * cursig
        # these are the boundaries in wavelength that will potentially
        # contribute to the current pixel

        left = np.searchsorted(lam0, curl0) - 1
        right = np.searchsorted(lam0, curl1)
        if left < 0:
            size_warning = True
            left = 0
        if right > len(lam0) - 2:
            size_warning = True
            right = len(lam0) - 2
        # I limit by the edges of the input spectrum

        curx = np.arange(left, right + 1)
        # these are pixel positions in the input template that
        # will be relevant

        x1 = lam0[curx]
        x2 = lam0[curx + 1]
        # these are neighboring pixels in the input template
        # we'll assume linear interpolation inbetween values on those

        l1 = curlam - leftstep
        l2 = curlam + rightstep
        # these are the edges of the pixel we will integrate over
        coeff1, coeff2 = pix_integrator(x1, x2, l1, l2, cursig)
        curstep = (leftstep + rightstep)
        ys.append(i + curx * 0)
        xs.append(curx)
        vals.append(coeff1 / curstep)

        ys.append(i + curx * 0)
        xs.append(curx + 1)
        vals.append(coeff2 / curstep)
        # we divide by the step to preserve the units of 'per wavelength'
    if size_warning:
        warnings.warn(
            'The input spectrum is not wide enough to do LSF convolution. '
            'The edges of the spectrum will be corrupted.')
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    vals = np.concatenate(vals)

    mat = scipy.sparse.coo_matrix((vals, (xs, ys)),
                                  shape=(len(lam0), len(lam)))
    mat = mat.tocsc()
    return mat


def apply_rebinner(mat, spec0):
    ret = np.array(spec0 @ mat)
    return ret


def rebin(lam0, spec0, newlam, resolution):
    """ Rebin a given spectrum lam0, spec0 to the new wavelength
    and new resolution

    Parameters
    ----------
    lam0: ndarray
        1d numpy array with wavelengths of the template pixels
    spec0: ndarray
        1d numpy array with the template spectrum
    newlam: ndarray
        1d array with the wavelengths of the output spectrum
    resolution: float
        Resolution lam/dlam

    Returns
    -------
    spec: ndarray
        Rebinned spectrum

    Example
    -------
    >>> lam,spec=read_grid.get_spec(dict(logg=1,teff=5250,feh=-1,alpha=0.4))
    >>> newlam = np.linspace(4000,9000,4000)
    >>> newspec=read_grid.rebin(lam, spec, newlam, 1800)
    """

    mat = make_rebinner(lam0, newlam, resolution)
    ret = apply_rebinner(mat, spec0)
    return ret


def main(args):
    parser = argparse.ArgumentParser(
        description='Create the database describing the templates')
    parser.add_argument('--prefix',
                        type=str,
                        default='./',
                        dest='prefix',
                        help='The location of the input grid')

    parser.add_argument('--keyword_teff',
                        type=str,
                        default='PHXTEFF',
                        help='The keyword for teff in the header')
    parser.add_argument('--keyword_logg',
                        type=str,
                        default='PHXLOGG',
                        help='The keyword for logg in the header')
    parser.add_argument('--keyword_alpha',
                        type=str,
                        default='PHXALPHA',
                        help='The keyword for alpha in the header')
    parser.add_argument('--keyword_metallicity',
                        type=str,
                        default='PHXM_H',
                        help='The keyword for metallicity in the header')
    parser.add_argument('--name_metallicity',
                        type=str,
                        default='feh',
                        help='The internal name for the metallicity')
    parser.add_argument('--name_alpha',
                        type=str,
                        default='alpha',
                        help='The internal name for the metallicity')

    parser.add_argument(
        '--extra_params',
        type=str,
        default=None,
        help='Extra template parameters in the form of comma separated '
        'value:value. I.e. if you want to store vmic values and they are '
        'in the VMIC keyword in the header use vmic:VMIC')
    parser.add_argument('--glob_mask',
                        type=str,
                        default='*/*fits',
                        help='The mask to find the spectra')

    parser.add_argument(
        '--templdb',
        type=str,
        default='files.db',
        help='The filename where the SQLite database describing the '
        'template library will be stored')
    args = parser.parse_args(args)
    keywords = dict(teff=args.keyword_teff, logg=args.keyword_logg)
    keywords[args.name_metallicity] = args.keyword_metallicity
    keywords[args.name_alpha] = args.keyword_alpha
    if args.extra_params is None:
        extra_params = None
    else:
        extra_params = args.extra_params.split(',')
        _tmp = {}
        for cur in extra_params:
            cur1, cur2 = cur.split(':')
            _tmp[cur1] = cur2
        extra_params = _tmp

    makedb(args.prefix,
           dbfile=args.templdb,
           keywords=keywords,
           mask=args.glob_mask,
           extra_params=extra_params,
           name_metallicity=args.name_metallicity,
           name_alpha=args.name_alpha)


if __name__ == '__main__':
    main(sys.argv[1:])

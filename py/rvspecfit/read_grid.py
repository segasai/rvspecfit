from __future__ import print_function
import glob
import sys
import sqlite3
import astropy.io.fits as pyfits
import scipy.stats
import scipy.sparse
import numpy as np
import astropy.wcs as pywcs
import argparse


def myintegrator(A, B, x1, x2, l1, l2, s):
    # this is the integral of (Ax+B)/sqrt(2pi)/s*exp(-1/2*(x-y)^2/s^2
    # for x=x1..x2 y=l1..l2
    # Here is the mathematica code
    # FortranForm[
    # Simplify[Integrate[(A*x + B)/Sqrt[2*Pi]/s*
    # Exp[-1/2*(x - y)^2/s^2], {x, x1, x2}, {y, l1, l2}]]]
    E = np.exp(1)
    Sqrt = np.sqrt
    Erf = scipy.special.erf
    Pi = np.pi
    return (-((Sqrt(2 / Pi) * s *
               (2 * B + A * (l1 + x1))) / E**((l1 - x1)**2 / (2. * s**2))) +
            (Sqrt(2 / Pi) * s * (2 * B + A * (l2 + x1))) / E**((l2 - x1)**2 /
                                                               (2. * s**2)) +
            (Sqrt(2 / Pi) * s * (2 * B + A * (l1 + x2))) / E**((l1 - x2)**2 /
                                                               (2. * s**2)) -
            (Sqrt(2 / Pi) * s *
             (2 * B + A * (l2 + x2))) / E**((l2 - x2)**2 / (2. * s**2)) + x1 *
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


def integrator(x1, x2, l1, l2, s):
    """
    Integrate the flux within the pixel given the lsf.
    We assume that the flux before LSF convolution is given by linear
    interpolation from x1,x2.
    The l1,l2 are scalar edges of the pixel within which we want to compute
    the flux. s is the sigma of the LSF
    The function returns two values of weights for y1,y2 where y1,y2 are 
    the values of the non-convolved spectra at x1,x2
    """
    ret1 = myintegrator(1 / (x1 - x2), -x2 / (x1 - x2), x1, x2, l1, l2, s)
    ret2 = myintegrator(1 / (x2 - x1), -x1 / (x2 - x1), x1, x2, l1, l2, s)
    return ret1, ret2


class ParamMapper:
    """
    Class used to map stellar atmospheric parameters into more manageable grid
    used for interpolation
    """
    def __init__(self):
        pass

    def forward(self, vec):
        """
        Map atmospheric parameters into parameters used in the grid for Interpolation
        That includes logarithming the teff
        
        Parameters
        -----------
        vec: numpy array
            The vector of atmospheric parameters Teff, logg, feh, alpha
        
        Returns
        ----------
        ret: numpy array
            The vector of transformed parameters used in interpolation
        """
        return np.array([np.log10(vec[0]), vec[1], vec[2], vec[3]])

    def inverse(self, vec):
        """
        Map transformed parameters used in the grid for interpolation back into
        the atmospheric parameters. That includes exponentiating the log10(teff)

        Parameters
        -----------
        vec: numpy array
            The vector of transformed atmospheric parameters log(Teff), logg, feh, alpha
        
        Returns
        ----------
        ret: numpy array
            The vector of original atmospheric parameters.
    """
        return np.array([10**vec[0], vec[1], vec[2], vec[3]])


def makedb(
    prefix='/physics2/skoposov/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/',
    dbfile='files.db'):
    """ Create an sqlite database of the templates """
    DB = sqlite3.connect(dbfile)
    id = 0
    DB.execute(
        'CREATE TABLE files (filename varchar, teff real, logg real, met real, alpha real, id int);'
    )

    mask = '*/*fits'
    fs = sorted(glob.glob(prefix + mask))
    if len(fs) == 0:
        raise Exception(
            "No FITS templates found in the directory specified (using mask %s"
            % mask)
    for f in fs:
        hdr = pyfits.getheader(f)
        teff = hdr['PHXTEFF']
        logg = float(hdr['PHXLOGG'])
        alpha = float(hdr['PHXALPHA'])
        met = float(hdr['PHXM_H'])

        DB.execute('insert into files  values (?, ? , ? , ? , ?, ? )',
                   (f.replace(prefix, ''), teff, logg, met, alpha, id))
        id += 1
    DB.commit()


def get_spec(
    logg,
    temp,
    met,
    alpha,
    dbfile='/tmp/files.db',
    prefix='/physics2/skoposov/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/',
    wavefile='/physics2/skoposov/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
):
    """ returns individual spectra
    > lam,spec=read_grid.get_spec(1,5250,-1,0.4)
    """
    deltalogg = 0.01
    deltatemp = 1
    deltaalpha = 0.01
    deltamet = 0.01

    query = '''select filename from files where
		teff between %f and %f and
		logg between %f and %f and
		alpha between %f and %f and
		met between %f and %f ''' % (
        temp - deltatemp, temp + deltatemp, logg - deltalogg, logg + deltalogg,
        alpha - deltaalpha, alpha + deltaalpha, met - deltamet, met + deltamet)

    conn = sqlite3.connect(dbfile)
    cur = conn.cursor()
    cur.execute(query)
    fnames = cur.fetchall()
    if len(fnames) > 1:
        print('Warning: More than 1 file returned', file=sys.stderr)
    if len(fnames) == 0:
        raise Exception('No spectra found')

    dat, hdr = pyfits.getdata(prefix + '/' + fnames[0][0], header=True)
    speclen = len(dat)
    lams = np.arange(speclen) * 1.
    lams = pyfits.getdata(wavefile)
    print('Using', fnames[0][0])
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
    l0 = len(lam0)
    l = len(lam)
    xs = []
    ys = []
    vals = []
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
        coeff1, coeff2 = integrator(x1, x2, l1, l2, cursig)
        curstep = (leftstep + rightstep)
        ys.append(i + curx * 0)
        xs.append(curx)
        vals.append(coeff1 / curstep)

        ys.append(i + curx * 0)
        xs.append(curx + 1)
        vals.append(coeff2 / curstep)
        # we divide by the step to preserve the units of 'per wavelength'

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    vals = np.concatenate(vals)

    mat = scipy.sparse.coo_matrix((vals, (xs, ys)),
                                  shape=(len(lam0), len(lam)))
    mat = mat.tocsc()
    return mat


def apply_rebinner(mat, spec0):
    ret = np.array(spec0 * mat)
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
    >>> lam,spec=read_grid.get_spec(1,5250,-1,0.4)
    >>> newlam = np.linspace(4000,9000,4000)
    >>> newspec=read_grid.rebin(lam, spec, newlam, 1800)
    """
    lam, spec = get_spec(1, 5000, -1, 0.2)

    mat = make_rebinner(lam0, newlam, resolution)
    ret = apply_rebinner(mat, spec0)
    return ret


def main(args):
    parser = argparse.ArgumentParser(
        description='Create the database descrbing the templates')
    parser.add_argument('--prefix',
                        type=str,
                        default='./',
                        dest='prefix',
                        help='The location of the input grid')
    parser.add_argument(
        '--templdb',
        type=str,
        default='files.db',
        help=
        'The filename where the SQLite database describing the template library will be stored'
    )
    args = parser.parse_args()
    makedb(args.prefix, dbfile=args.templdb)


if __name__ == '__main__':
    main(sys.argv[1:])

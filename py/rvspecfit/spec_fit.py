import sys
if (sys.version_info < (3, 0)):
    import functools32 as functools
else:
    import functools
import random
import numpy as np
import numpy.random
import scipy
import scipy.interpolate
from scipy.constants.constants import speed_of_light
import scipy.sparse
import scipy.signal
import collections

from rvspecfit import frozendict
from rvspecfit import utils
from rvspecfit import spec_inter
from rvspecfit import cubic


class LRUDict:
    """ Simple implementation of LRU dictionary """
    def __init__(self, N):
        self.N = N
        self.D = collections.OrderedDict()

    def __contains__(self, x):
        return x in self.D

    def __setitem__(self, x, y):
        inside = x in self.D
        if len(self.D) == self.N and not inside:
            del self.D[next(iter(self.D.keys()))]
        if inside:
            self.D.move_to_end(x, last=True)
        self.D[x] = y

    def __getitem__(self, x):
        ret = self.D[x]
        self.D.move_to_end(x, last=True)
        return ret

    def __str__(self):
        return self.D.__str__()


# resolution matrix
class ResolMatrix:
    def __init__(self, mat):
        self.fd = {}
        self.fd['mat'] = mat
        # id of the object to ensure that I can cache calls on a given data
        self.id = random.getrandbits(128)

    def __hash__(self):
        return self.id

    @property
    def mat(self):
        return self.fd['mat']


class SpecData:
    '''
    Class describing a single spectrocopic dataset
    '''
    def __init__(self, name, lam, spec, espec, badmask=None):
        '''
        Construct the spectroscopic dataset

        Parameters
        -----------
        name: string
            Name of the spectroscopic setups
        lam: numpy array
            Wavelength vector
        spec: numpy array
            Vector with the spectrum
        espec: numpy array
            Vector with the error spectrum (sigmas)
        badmask: numpy array (boolean, optional)
            The mask with bad pixels
        '''
        self.fd = {}
        self.fd['name'] = name
        self.fd['lam'] = lam
        self.fd['spec'] = spec
        self.fd['espec'] = espec
        if badmask is None:
            badmask = np.zeros(len(spec), dtype=bool)
        self.fd['badmask'] = badmask
        self.fd = utils.freezeDict(self.fd)
        # id of the object to ensure that I can cache calls on a given data
        self.id = random.getrandbits(128)

    @property
    def name(self):
        return self.fd['name']

    @property
    def lam(self):
        return self.fd['lam']

    @property
    def spec(self):
        return self.fd['spec']

    @property
    def espec(self):
        return self.fd['espec']

    @property
    def badmask(self):
        return self.fd['badmask']

    def __hash__(self):
        return self.id


@functools.lru_cache(100)
def get_basis(specdata, npoly, rbf=True):
    '''Get the precomputed polynomials for the continuum for a given specdata

    Parameters
    ----------

    specdata: SpecData objects
        The spectroscopic dataset objects
    npoly: integer
        The degree of polynomial to use
    rbf: bool
        Use the RBF basis instead of monomial basis

    Returns
    -------
    polys: numpy array(npolys, Nwave)
        The array of continuum polynomials

    '''
    lam = specdata.lam
    # get polynomials for continuum
    polys = np.zeros((npoly, len(lam)))
    normlam = (lam - lam[0]) / (lam[-1] - lam[0]) * 2 - 1
    # -1..1
    if not rbf:
        coeffs = {}
        for i in range(npoly):
            coeffs[i] = np.zeros(npoly)
            coeffs[i][i] = 1
        for i in range(npoly):
            polys[i, :] = np.polynomial.Chebyshev(coeffs[i])(normlam)
    else:
        npoly0 = 3
        for i in range(npoly0):
            polys[i, :] = normlam**i
        nrbf = npoly - npoly0
        assert (npoly >= npoly0)
        if nrbf > 0:
            sig = 1. / nrbf
            # larger values lead to
            # poorly conditioned matrices and noisy likelihood
            # BE CAREFUL
            rbfcens = np.linspace(-1, 1, nrbf, True)
            for i in range(nrbf):
                polys[npoly0 + i, :] = np.exp(
                    -0.5 * (normlam - rbfcens[i])**2 / sig**2)

    return polys


def get_chisq0(spec, templ, polys, get_coeffs=False, espec=None):
    '''
    Get the chi-square values for the vector of velocities and grid of
    templates after marginalizing over continuum
    If espec is not provided we assume data and template are alreay normalized
    Importantly the returned chi-square is not the true chi-square, but instead
    the -2*log(L)

    Parameters
    ----------

    spec: numpy
        Spectrum array
    templ: numpy
        The template
    polys: numpy
        The continuum polynomials
    get_coeffs: boolean (optional)
        If true return the coefficients of polynomials
    espec: numpy (optional)
        If specified, this is the error vector. If not specified, then it is
        assumed that spectrum and template are already divided by the
        uncertainty

    Returns
    --------
    chisq: real
        Chi-square of the data
    coeffs: numpy
        The polynomial coefficients (optional)

    '''

    if espec is not None:
        normspec = spec / espec
        normtempl = templ / espec
    else:
        normspec = spec
        normtempl = templ

    polys1 = normtempl[None, :] * polys
    # M
    vector1 = polys1 @ normspec
    # M^T S

    matrix1 = np.dot(polys1, polys1.T)
    u, s, v = scipy.linalg.svd(matrix1, check_finite=False)
    det = np.prod(s)
    # matrix1 = usv
    # matrix1 = u @ np.diag(s**2) @ u.T
    # matrix1 is the M^T M matrix

    v2 = v.T @ (
        (1. / s)[:, None] * u.T) @ vector1  # this is matrix1^(-1) vector1

    chisq = -vector1.T @ v2 + \
        0.5 * np.log(det)
    if get_coeffs:
        coeffs = v2.flatten()
        return chisq, coeffs
    else:
        return chisq


@functools.lru_cache(100)
def getCurTempl(spec_setup, atm_param, rot_params, config):
    """
    Get the spectrum in the given setup with given atmospheric parameters and
    given config

    Parameters
    -----------
    spec_setup: string
        The name of the spectroscopic setup
    atm_param: tuple
        The atmospheric parameters
    rot_params: tuple
        The parameters of stellar rotation models (could be None)
    resol_params: object
        The object that descibe the resolution convolution (could be None)
    config: dict
        The configuration dictionary

    Returns
    templ: numpy
        The template vector
    """
    curInterp = spec_inter.getInterpolator(spec_setup, config)
    outside = float(curInterp.outsideFlag(atm_param))
    spec = curInterp.eval(atm_param)
    if not np.isfinite(outside):
        # The spectrum may be completely crap
        pass
    else:
        # take into account the rotation of the star
        if rot_params is not None:
            spec = convolve_vsini(curInterp.lam, spec, *rot_params)

    templ_tag = random.getrandbits(128)
    return outside, curInterp.lam, spec, templ_tag


def construct_resol_mat(lam, resol=None, width=None):
    '''Construct a sparse resolution matrix from a resolution number R

    Parameters
    ----------
    lam: numpy
        The wavelength vector
    resol: real/numpy
        The resolution value (R=lambda/delta lambda)
    width: real
        The Gaussian width of the kernel in angstrom
        (cannot be specified together with resol)

    Returns
    -------
    mat: scipy.sparse matrix
        The matrix describing the resolution convolution operation

    '''
    assert (resol is None or width is None)
    assert (resol is not None or width is not None)
    if resol is not None:
        sigs = lam / resol / 2.35
    else:
        if np.isscalar(width):
            sigs = np.zeros(len(lam)) + width
        else:
            sigs = width
    thresh = 5
    assert (np.all(np.diff(lam) > 0))
    l1 = lam - thresh * sigs
    l2 = lam + thresh * sigs
    # leftmost/rightmost wavelength edges contributing to the convolution

    i1 = np.searchsorted(lam, l1, 'left')
    i1 = np.maximum(i1, 0)
    # pixelids of leftmost edges, truncate at 0

    i2 = np.searchsorted(lam, l2, 'right')
    i2 = np.minimum(i2, len(lam) - 1)
    # pixelids of rightmost edges, truncate at right edge

    lampix = np.arange(len(lam))
    maxl = max(np.max(i2 - lampix), np.max(lampix - i1))
    maxl = min(len(lam), maxl)
    # maximum number of pixels to use

    offsets = np.arange(-maxl, maxl + 1)
    xs2d = lampix[None, :] + offsets[:, None]
    # 2d array of pixels of neighbors shape (win, npix)

    mask = (xs2d >= 0) & (xs2d < len(lam))
    xs2d[~mask] = 0
    xs2d[~mask] = 0
    # zero-out outside boundary ones

    XL = np.exp(-0.5 * ((lam[xs2d] - lam[None, :]) / sigs[None, :])**2) * mask
    XL = XL / XL.sum(axis=0)[None, :]
    yids = (lampix[None, :] + (len(lam) - offsets)[:, None]) % len(lam)
    xids = yids * 0 + maxl + offsets[:, None]
    XL = XL[xids, yids]
    mat = scipy.sparse.spdiags(XL, offsets, len(lam), len(lam))
    return ResolMatrix(mat)


def convolve_resol(spec, resol_matrix):
    '''Convolve the spectrum with the resolution matrix

    Parameters
    ----------

    spec: numpy
        The spectrum array
    resol_matrix: ResolMatrix object
        The resolution matrix object

    Returns
    -------

    spec: numpy
        The spectrum array

    '''
    return resol_matrix.mat @ spec


def rotation_kernel(x):
    eps = 0.6  # limb darkening coefficient
    # See https://ui.adsabs.harvard.edu/abs/2011A%26A...531A.143D/abstract
    # x = ln(lam/lam0) *c/vsini
    return (2 * (1 - eps) * np.sqrt(1 - x**2) + np.pi / 2 * eps *
            (1 - x**2)) / 2 / np.pi / (1 - eps / 3)


def convolve_vsini(lam_templ, templ, vsini):
    """Convolve the spectrum with the stellar rotation velocity kernel

    Parameters
    ----------

    lam_templ: numpy
        The wavelength vector (MUST be spaced logarithmically)
    templ: numpy
        The spectrum vector
    vsini: real
        The Vsini velocity

    Returns
    -------
    spec: numpy
        The convolved spectrum

    """
    if vsini == 0:
        return templ
    lnstep = np.log(lam_templ[1] / lam_templ[0])
    amp = vsini * 1e3 / speed_of_light
    npts = np.ceil(amp / lnstep)
    xgrid = np.arange(-npts, npts + 1) * lnstep / amp
    good = np.abs(xgrid) <= 1
    kernel = xgrid * 0.
    kernel[good] = rotation_kernel(xgrid[good])
    # ensure that the lambda is spaced logarithmically
    assert (np.allclose(lam_templ[1] / lam_templ[0],
                        lam_templ[-1] / lam_templ[-2]))
    templ1 = scipy.signal.fftconvolve(templ, kernel, mode='same')
    return templ1


def getRVInterpol(lam_templ, templ):
    """
    Produce the spectrum interpolator to evaluate the spectrum at arbitrary
    wavelengths

    Parameters
    -----------
    lam_templ: numpy
        Wavelength array
    templ: numpy
        Spectral array

    Returns
    --------
    interpol: scipy.interpolate object
        The object that can be used to evaluate template at any wavelength
    """

    interpol = cubic.CubicSpline(lam_templ, templ, extrapolate=False)
    return interpol


def evalRV(interpol, vel, lams):
    """
    Evaluate the spectrum interpolator at a given velocity and given
    wavelengths

    Parameters
    -----------
    interpol: scipy.intepolate object
        Template interpolator
    vel: real
        Radial velocity
    lams: numpy
        Wavelength array

    Returns
    --------
    spec: numpy
        Evaluated spectrum
    """
    beta = vel * 1000. / speed_of_light
    return interpol(lams * np.sqrt((1 - beta) / (1 + beta)))


def param_dict_to_tuple(paramDict, setup, config):
    # convert the dictionary with spectral parameters
    # to a tuple
    # addutional arguments are spectral setup
    # and configuration object
    interpolator = spec_inter.getInterpolator(setup, config)
    return tuple([paramDict[_] for _ in interpolator.parnames])


def get_chisq_continuum(specdata, options=None):
    '''
    Fit the spectrum with continuum only

    Parameters
    ----------

    specdata: list of Specdata
        Input spectra
    options: dict
        Dictionary of options (npoly option is required)

    Returns
    -------
    ret: list
        Array of chi-squares

    '''
    npoly = options.get('npoly') or 5
    rbf = options.get('rbf_continuum') or True
    ret = []
    ret1 = []
    for curdata in specdata:
        # name = curdata.name
        polys = get_basis(curdata, npoly, rbf=rbf)
        templ = np.ones(len(curdata.spec))
        curchisq, coeffs = get_chisq0(curdata.spec,
                                      templ,
                                      polys,
                                      get_coeffs=True,
                                      espec=curdata.espec)
        model = np.dot(coeffs, polys * templ)
        curchisq = (((model - curdata.spec) / curdata.espec)**2).sum()
        redchisq = curchisq / len(curdata.espec)
        ret.append(curchisq)
        ret1.append(redchisq)
    ret = np.array(ret)
    ret1 = np.array(ret1)
    return dict(chisq_array=ret, redchisq_array=ret1)


def get_chisq(specdata,
              vel,
              atm_params,
              rot_params,
              resol_params,
              options=None,
              config=None,
              cache=None,
              full_output=False):
    """ Find the chi-square of the dataset at a given velocity
        atmospheric parameters, rotation parameters
        and resolution parameters

    Parameters
    ----------

    specdata: spec_fit.SpecData
        The object with the data to be fitted
    vel: real
        The radial velocity
    atm_params: tuple
        The tuple with parameters of the star
    rot_parameters: tuple
        The tuple with parameters of the rotation (can be None)
    resol_parameters: tuple
        The parameters of the spectral resolution (can be None)
    options: dict
        The dictionary with fitting options (such as npoly for the degree of
        the polynomial)
    config: dict (optional)
        The configuration objection
    cache: dict (optional)
        The cache object, to preserve info between calls
    full_output: bool
        If full_output is set more info is returned

    Returns
    -------
    ret: float or dictionary
        If full_output is False, ret is float = -2*log(L) of the whole data
        If full_output is True ret is a dictionary with the following keys
        chisq -- this is the -2*log(L) of the whole dataset
        chisq_array -- this is the array of chi-squares (proper ones)
        for each of the fited spectra
        redchisq_array -- this is the array of reduced chi-squares
        models -- array of best fit models
        raw_models -- array of models not corrected by the polynomial 

    """
    npoly = options.get('npoly') or 5
    rbf = options.get('rbf_continuum') or True
    logl = 0
    badchi = 1e6
    if rot_params is not None:
        rot_params = tuple(rot_params)
    if resol_params is not None:
        resol_params = frozendict.frozendict(resol_params)
    atm_params = tuple(atm_params)

    models = []
    raw_models = []
    chisq_array = []
    red_chisq_array = []
    # iterate over multiple datasets
    for curdata in specdata:
        name = curdata.name

        outside, templ_lam, templ_spec, templ_tag = getCurTempl(
            name, atm_params, rot_params, config)

        # if the current point is outside the template grid
        # add bad value and bail out

        if not np.isfinite(outside):
            logl += badchi
            chisq_array.append(np.nan)
            red_chisq_array.append(np.nan)
            models.append(np.zeros(len(curdata.lam)) + np.nan)
            continue
        else:
            logl += outside * badchi

        if (curdata.lam[0] < templ_lam[0] or curdata.lam[0] > templ_lam[-1]
                or curdata.lam[-1] < templ_lam[0]
                or curdata.lam[-1] > templ_lam[-1]):
            raise Exception(
                ("The template library (%f,%f)  doesn't cover" +
                 "this wavelength range (%f,%f)") %
                (templ_lam[0], templ_lam[-1], curdata.lam[0], curdata.lam[-1]))

        # current template interpolator object
        if cache is None or templ_tag not in cache:
            curtemplI = getRVInterpol(templ_lam, templ_spec)
            if cache is not None:
                cache[templ_tag] = curtemplI
        else:
            curtemplI = cache[templ_tag]

        evalTempl = evalRV(curtemplI, vel, curdata.lam)

        # take into account the resolution
        if resol_params is not None:
            evalTempl = convolve_resol(evalTempl, resol_params[name])

        polys = get_basis(curdata, npoly, rbf=rbf)

        curlogl = get_chisq0(curdata.spec,
                             evalTempl,
                             polys,
                             get_coeffs=full_output,
                             espec=curdata.espec)
        if full_output:
            curlogl, coeffs = curlogl
            curmodel = np.dot(coeffs, polys * evalTempl)
            raw_models.append(evalTempl)
            models.append(curmodel)
            XCHISQ = (((curmodel - curdata.spec) / curdata.espec)**2).sum()
            chisq_array.append(XCHISQ)
            red_chisq_array.append(XCHISQ / len(curdata.espec))

        assert (np.isfinite(curlogl.item()))
        logl += np.asscalar(curlogl.item())

    if full_output:
        ret = {}
        ret['chisq'] = logl
        # chisq here is the -2*log(L)
        ret['chisq_array'] = chisq_array
        ret['red_chisq_array'] = red_chisq_array
        ret['models'] = models
        ret['raw_models'] = raw_models
    else:
        ret = logl
    return ret


def quadratic_interp_min(vel_grid, chisq, i):
    """Find the minimum using quadratic interpolation

    Parameters
    -----------
    vel_grid: numpy
        Array of velocities
    chisq: numpy
        Array of chi-squares
    i: int
        Index of the point with the smallest chisq

    Returns
    --------
    The estimated of velocity minimum
    """
    if i == 0 or i == len(vel_grid) - 1:
        return vel_grid[i]
    x = vel_grid[i - 1:i + 2]
    y = chisq[i - 1:i + 2]
    a2, a1, a0 = np.polyfit(x, y, 2)
    val = -a1 / 2 / a2
    assert ((val < vel_grid[i + 1]) & (val > vel_grid[i - 1]))
    return val


def find_best(specdata,
              vel_grid,
              params_list,
              rot_params,
              resol_params,
              options=None,
              config=None,
              quadratic=True):
    """ Find the best fit template and velocity from a grid

    Parameters
    ----------
    specdata: list of SpecData
        Spectroscopic dataset
    vel_grid: ndarray
        Array of velocities
    params_list: list
        Array of parameters to explore
    rot_params: tuple
        Parameters of rotation (or None)
    resol_params: tuple
        Parameters of resolution convolution (or None)
    options: dict
        Dictionary of options
    config: dict
        Dictionary of the configuration
    quadratic: bool
        if True we try to do velocity quadratic interpolation near the max

    Returns
    -------
    ret: dict
        Dictionary with measured parameters. The keys are
        best_vel -- velocity
        vel_err -- velocity error
        best_param -- best param
        kurtosis -- kurtosi of velocity distribution
        skewness -- skewness of velocity distribution
        probs -- vector of 'posterior' probabilities over the grid

    """
    cache = LRUDict(100)
    chisq = np.zeros((len(vel_grid), len(params_list)))
    for j, curparam in enumerate(params_list):
        for i, v in enumerate(vel_grid):
            chisq[i, j] = get_chisq(specdata,
                                    v,
                                    curparam,
                                    rot_params,
                                    resol_params,
                                    options=options,
                                    config=config,
                                    cache=cache)
    xind = np.argmin(chisq)
    i1, i2 = np.unravel_index(xind, chisq.shape)
    probs = np.exp(-0.5 * (chisq[:, i2] - chisq[i1, i2]))
    probs = probs / probs.sum()
    if quadratic:
        best_vel = quadratic_interp_min(vel_grid, chisq[:, i2], i1)
    else:
        best_vel = vel_grid[i1]
    best_err = np.sqrt((probs * (vel_grid - best_vel)**2).sum())
    if best_err < 1e-10:
        kurtosis, skewness = 0, 0
    else:
        kurtosis = ((probs * (vel_grid - best_vel)**4).sum()) / best_err**4
        skewness = ((probs * (vel_grid - best_vel)**3).sum()) / best_err**3
    return dict(best_chi=chisq[i1, i2],
                best_vel=best_vel,
                vel_err=best_err,
                best_param=params_list[i2],
                kurtosis=kurtosis,
                skewness=skewness,
                probs=probs)

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
            self.D.move_to_end(x,last=True)
        self.D[x] = y

    def __getitem__(self,x):
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

        Parameters:
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
def get_polys(specdata, npoly):
    '''
    Get the precomputed polynomials for the continuum for a given specdata

    Parameters:
    -----------
    specdata: SpecData objects
        The spectroscopic dataset objects
    npoly: integer
        The degree of polynomial to use

    Returns:
    --------
    polys: numpy array(npolys, Nwave)
        The array of continuum polynomials
    '''
    lam = specdata.lam
    # get polynomials for continuum
    polys = np.zeros((npoly, len(lam)))
    coeffs = {}
    for i in range(npoly):
        coeffs[i] = np.zeros(npoly)
        coeffs[i][i] = 1
    normlam = (lam - lam[0]) / (lam[-1] - lam[0]) * 2 - 1
    # -1..1
    for i in range(npoly):
        polys[i, :] = np.polynomial.Chebyshev(coeffs[i])(normlam)
    return polys


def get_chisq0(spec, templ, polys, get_coeffs=False, espec=None):
    '''
    Get the chi-square values for the vector of velocities and grid of templates
    after marginalizing over continuum
    If espec is not provided we assume data and template are alreay normalized

    Parameters:
    -----------
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
        assumed that spectrum and template are already divided by the uncertainty

    Returns:
    --------
    chisq: real
        Chi-square of the fitVsini
    coeffs: numpy
        The polynomial coefficients (optional)
    '''

    if espec is not None:
        normspec = spec / espec
        normtempl = templ / espec
    else:
        normspec = spec
        normtempl = templ

    npoly = polys.shape[0]
    matrix1 = np.matrix(np.zeros((npoly, npoly)), copy=False)
    #vector1 = np.matrix(np.zeros((npoly, 1)), copy=False)
    polys1 = np.zeros((npoly, len(spec)))

    polys1[:, :] = normtempl[None, :] * polys
    # M
    vector1 = np.matrix(np.dot(polys1, normspec)).T
    # M^T S
    u, s, v = scipy.linalg.svd(polys1, full_matrices=False, check_finite=False)
    u = np.matrix(u, copy=False)
    matrix1 = u * np.matrix(np.diag(s**2)) * u.T
    det = np.prod(s)**2
    # matrix1 is the M^T M matrix
    v2 = scipy.linalg.solve(matrix1, vector1, check_finite=False)

    chisq = -vector1.getT() * v2 + \
        0.5 * np.log(det)
    if get_coeffs:
        coeffs = v2.flatten()
        return chisq, coeffs
    else:
        return chisq


@functools.lru_cache(100)
def getCurTempl(spec_setup, atm_param, rot_params, resol_params, config):
    """
    Get the spectrum in the given setup with given atmospheric parameters and
    given config

    Parameters:
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

    Returns:
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


def construct_resol_mat(lam, R):
    '''
    Construct a sparse resolution matrix from a resolution number R

    Parameters:
    lam: numpy
        The wavelength vector
    R: real/numpy
        The resolution value (R=lambda/delta lambda)

    Returns:
    mat: scipy.sparse matrix
        The matrix describing the resolution convolution operation
    '''
    sigs = lam / R / 2.35
    thresh = 5
    assert (np.all(np.diff(lam) > 0))
    l1 = lam - thresh * sigs
    l2 = lam + thresh * sigs
    i1 = np.searchsorted(lam, l1, 'left')
    i1 = np.maximum(i1, 0)
    i2 = np.searchsorted(lam, l2, 'right')
    i2 = np.minimum(i2, len(lam) - 1)
    xs = []
    ys = []
    vals = []
    for i in range(len(lam)):
        iis = np.arange(i1[i], i2[i] + 1)
        kernel = np.exp(-0.5 * (lam[iis] - lam[i])**2 / sigs[i]**2)
        kernel = kernel / kernel.sum()
        ys.append(iis)
        xs.append([i] * len(iis))
        vals.append(kernel)
    xs, ys, vals = [np.concatenate(_) for _ in [xs, ys, vals]]
    mat = scipy.sparse.coo_matrix((vals, (xs, ys)), shape=(len(lam), len(lam)))
    mat = mat.tocsc()
    return ResolMatrix(mat)


def convolve_resol(spec, resol_matrix):
    '''
    Convolve the spectrum with the resolution matrix

    Parameters:
    spec: numpy
        The spectrum array
    resol_matrix: ResolMatrix object
        The resolution matrix object

    Returns:
    --------
    spec: numpy
        The spectrum array
    '''
    return resol_matrix.mat * spec


def convolve_vsini(lam_templ, templ, vsini):
    """
    Convolve the spectrum with the stellar rotation velocity kernel

    Parameters:
    lam_templ: numpy
        The wavelength vector (MUST be spaced logarithmically)
    templ: numpy
        The spectrum vector
    vsini: real
        The Vsini velocity

    Returns:
    --------
    spec: numpy
        The convolved spectrum
    """
    eps = 0.6  # limb darkening coefficient

    def kernelF(x):
        return (2 * (1 - eps) * np.sqrt(1 - x**2) + np.pi / 2 * eps *
                (1 - x**2)) / 2 / np.pi / (1 - eps / 3)

    step = np.log(lam_templ[1] / lam_templ[0])
    amp = vsini * 1e3 / speed_of_light
    npts = np.ceil(amp / step)
    xgrid = np.arange(-npts, npts + 1) * step
    kernel = kernelF(xgrid)
    kernel[np.abs(xgrid) > 1] = 0
    # ensure that the lambda is spaced logarithmically
    assert (np.allclose(lam_templ[1] / lam_templ[0],
                        lam_templ[-1] / lam_templ[-2]))
    templ1 = scipy.signal.fftconvolve(templ, kernel, mode='same')
    return templ1


def getRVInterpol(lam_templ, templ):
    """
    Produce the spectrum interpolator to evaluate the spectrum at arbitrary
    wavelengths

    Parameters:
    -----------
    lam_templ: numpy
        Wavelength array
    templ: numpy
        Spectral array

    Returns:
    --------
    interpol: scipy.interpolate object
        The object that can be used to evaluate template at any wavelength
    """

    interpol = scipy.interpolate.UnivariateSpline(
        lam_templ, templ, s=0, k=3, ext=2)
    return interpol


def evalRV(interpol, vel, lams):
    """
    Evaluate the spectrum interpolator at a given velocity and given wavelengths

    Parameters:
    -----------
    interpol: scipy.intepolate object
        Template interpolator
    vel: real
        Radial velocity
    lams: numpy
        Wavelength array

    Returns:
    --------
    spec: numpy
        Evaluated spectrum
    """
    return interpol(lams / (1 + vel * 1000. / speed_of_light))


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
    
    Parameters:
    specdata: list of Specdata
        Input spectra
    options: dict
        Dictionary of options (npoly option is required)

    Returns:
    --------
    ret: list
        Array of chi-squares
    '''
    npoly = options.get('npoly') or 5
    ret = []
    for curdata in specdata:
        name = curdata.name
        polys = get_polys(curdata, npoly)
        templ = np.ones(len(curdata.spec))
        curchisq, coeffs = get_chisq0(
            curdata.spec, templ, polys, get_coeffs=True, espec=curdata.espec)
        model = np.dot(coeffs, polys * templ)
        curchisq = (((model - curdata.spec) / curdata.espec)**2).mean()
        ret.append(curchisq)
    return ret


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
    """
    npoly = options.get('npoly') or 5
    chisq = 0
    outsides = 0
    models = []
    badchi = 1e6
    if rot_params is not None:
        rot_params = tuple(rot_params)
    if resol_params is not None:
        resol_params = frozendict.frozendict(resol_params)
    atm_params = tuple(atm_params)

    chisq_array = []
    # iterate over multiple datasets
    for curdata in specdata:
        name = curdata.name

        outside, templ_lam, templ_spec, templ_tag = getCurTempl(
            name, atm_params, rot_params, resol_params, config)

        # if the current point is outside the template grid
        # add bad value and bail out

        if not np.isfinite(outside):
            chisq += badchi
            chisq_array.append(np.nan)
            continue
        else:
            chisq += outside

        if (curdata.lam[0] < templ_lam[0] or curdata.lam[0] > templ_lam[-1]
                or curdata.lam[-1] < templ_lam[0]
                or curdata.lam[-1] > templ_lam[-1]):
            raise Exception(
                "The template library doesn't cover this wavelength")

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

        polys = get_polys(curdata, npoly)

        curchisq = get_chisq0(
            curdata.spec,
            evalTempl,
            polys,
            get_coeffs=full_output,
            espec=curdata.espec)
        if full_output:
            curchisq, coeffs = curchisq
            curmodel = np.dot(coeffs, polys * evalTempl)
            models.append(curmodel)
            chisq_array.append((((curmodel - curdata.spec) / curdata.espec)
                                **2).mean())
        assert (np.isfinite(np.asscalar(curchisq)))
        chisq += np.asscalar(curchisq)

    if full_output:
        ret = {}
        ret['chisq'] = chisq
        ret['chisq_array'] = chisq_array
        ret['models'] = models
    else:
        ret = chisq
    return ret


def find_best(specdata,
              vel_grid,
              params_list,
              rot_params,
              resol_params,
              options=None,
              config=None):
    # find the best fit template and velocity from a grid
    cache = LRUDict(100)
    chisq = np.zeros((len(vel_grid), len(params_list)))
    for j, curparam in enumerate(params_list):
        for i, v in enumerate(vel_grid):
            chisq[i, j] = get_chisq(
                specdata,
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
    best_vel = vel_grid[i1]
    best_err = np.sqrt((probs * (vel_grid - best_vel)**2).sum())
    if best_err < 1e-10:
        kurtosis, skewness = 0, 0
    else:
        kurtosis = ((probs * (vel_grid - best_vel)**4).sum()) / best_err**4
        skewness = ((probs * (vel_grid - best_vel)**3).sum()) / best_err**3
    return dict(
        best_chi=chisq[i1, i2],
        best_vel=vel_grid[i1],
        vel_err=best_err,
        best_param=params_list[i2],
        kurtosis=kurtosis,
        skewness=skewness,
        probs=probs)

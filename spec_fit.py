import astropy.io.fits as pyfits
import functools
import scipy
import joblib
import numpy as np
import scipy.interpolate
import numpy.random
import functools
from scipy.constants.constants import speed_of_light
import scipy.signal
import gc
import os
import math
import spec_inter
import yaml
from tempfile import mkdtemp
import frozendict
import random
import pylru
import scipy.sparse


def freezeDict(d):
    if isinstance(d, dict):
        d1 = {}
        for k, v in d.items():
            d1[k] = freezeDict(v)
        return frozendict.frozendict(d1)
    else:
        return d

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
    def __init__(self, name, lam, spec, espec, badmask=None):
        self.fd = {}
        self.fd['name'] = name
        self.fd['lam'] = lam
        self.fd['spec'] = spec
        self.fd['espec'] = espec
        self.fd['badmask'] = np.zeros(len(spec), dtype=bool)
        self.fd = freezeDict(self.fd)
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


def eval_vel_grid(vels_grid, interpol, lam_object):
    # obtain the interpolators for each template in the list
    interpols_grid = np.zeros((len(vels_grid), len(lam_object)))
    for j_vel, vel in enumerate(vels_grid):
        interpols_grid[j_vel, :] = interpol(lam_object /
                                            (1 + vel * 1000. / speed_of_light))
    return interpols_grid


@functools.lru_cache(100)
def get_polys(specdata, npoly):
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


def get_findmin(chisqs, vels):
    # find the mininum in chi-square
    # and fit the parabola around in the minimum
    minpos = np.argmin(chisqs)
    left = max(minpos - 2, 0)
    right = min(minpos + 3, len(chisqs) - 1)
    if minpos >= 1 or minpos < (len(chisqs) - 1):
        a, b, c = scipy.polyfit(vels[left:right],
                                chisqs[left:right], 2)
        evel = 1 / np.sqrt(a)
        bestvel = -b / 2 / a
    else:
        bestvel = chisqs[minpos]
        evel = vels[1] - vels[0]
    return (bestvel, evel, chisqs[minpos])


def get_chisq0(spec, templ, polys, getCoeffs=False, espec=None):
    # get the chi-square values for the vector of velocities
    # and grid of templates
    # after marginalizing over continuum
    # if espec is not provided we assume data and template are already
    # normalized
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
    if getCoeffs:
        coeffs = v2.flatten()
        return chisq, coeffs
    else:
        return chisq


def get_chisq_many(spec, espec, templgrid, polys):
    # get the chi-square values for the vector of velocities
    # and grid of templates
    assert(templgrid.shape == 2)

    chisqs = np.zeros(templgrid.shape[0])
    normspec = spec / espec
    normgrid = templgrid / espec[None, :]
    for i in range(templgrid.shape[0]):
        chisqs[i] = get_chisq(normspec, normgrid[i],
                              polys, getCoeffs=False, espec=None)
    return chisqs


@functools.lru_cache(100)
def getCurTempl(name, atm_param, rot_params, resol_params, config):
    """ get the spectrum in the given setup
    with given atmospheric parameters and given config
    """
    curInterp = spec_inter.getInterpolator(name, config)
    outside = curInterp.outsideFlag(atm_param)
    spec = curInterp.eval(atm_param)

    # take into account the rotation of the star
    if rot_params is not None:
        spec = convolve_vsini(curInterp.lam, spec, *rot_params)

    templ_tag = random.getrandbits(128)
    return outside, curInterp.lam, spec, templ_tag


def construct_resol_mat(lam, R):
    " Construct a sparse resolution matrix "
    sigs = lam / R / 2.35
    thresh = 5
    assert(np.all(np.diff(lam) > 0))
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
    return ResolMatrix(mat)


def convolve_resol(spec, resol_matrix):
    return resol_matrix.mat * spec


def convolve_vsini(lam_templ, templ, vsini):
    """ convolve the spectrum with the stellar rotation velocity kernel
    """
    eps = 0.6  # limb darkening coefficient

    def kernelF(x): return (2 * (1 - eps) * np.sqrt(1 - x**2) +
                            np.pi / 2 * eps * (1 - x**2)) / 2 / np.pi / (1 - eps / 3)
    step = np.log(lam_templ[1] / lam_templ[0])
    amp = vsini * 1e3 / speed_of_light
    npts = np.ceil(amp / step)
    xgrid = np.arange(-npts, npts + 1) * step
    kernel = kernelF(xgrid)
    kernel[np.abs(xgrid) > 1] = 0
    # ensure that the lambda is spaced logarithmically
    assert(np.allclose(lam_templ[1] / lam_templ[0],
                       lam_templ[-1] / lam_templ[-2]))
    templ1 = scipy.signal.fftconvolve(templ, kernel, mode='same')
    #raise Exception("not implemented yet")
    return templ1


def getRVInterpol(lam_templ, templ):
    """ Return the spectrum interpolator"""
    interpol = scipy.interpolate.UnivariateSpline(
        lam_templ, templ, s=0, k=3, ext=2)
    return interpol


def evalRV(interpol, vel, lams):
    """ Evaluate the spectrum interpolator at a given velocity
    and given wavelengths
    """
    return interpol(lams / (1 + vel * 1000. / speed_of_light))


def read_config(fname=None):
    if fname is None:
        fname = 'config.yaml'
    with open(fname) as fp:
        return freezeDict(yaml.safe_load(fp))


def param_dict_to_tuple(paramDict, setup, config):
    # convert the dictionary with spectral parameters
    # to a tuple
    # addutional arguments are spectral setup
    # and configuration object
    interpolator = spec_inter.getInterpolator(setup, config)
    return tuple([paramDict[_] for _ in interpolator.parnames])


def get_chisq(specdata, vel, atm_params, rot_params, resol_params, options=None,
              config=None, getModel=False, cache=None):
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

    # iterate over multiple datasets
    for curdata in specdata:
        name = curdata.name

        outside, templ_lam, templ_spec, templ_tag = getCurTempl(
            name, atm_params, rot_params,
            resol_params, config)

        # if the current point is outside the template grid
        # add bad value and bail out

        outsides += np.asscalar(outside)
        if not np.isfinite(outside):
            chisq += badchi
            continue

        if (curdata.lam[0] < templ_lam[0] or curdata.lam[0] > templ_lam[-1] or
                curdata.lam[-1] < templ_lam[0] or curdata.lam[-1] > templ_lam[-1]):
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
            evalTempl = convolve_resol(
                evalTempl, resol_params[name])

        polys = get_polys(curdata, npoly)

        curchisq = get_chisq0(curdata.spec, evalTempl,
                              polys, getCoeffs=getModel, espec=curdata.espec)
        if getModel:
            curchisq, coeffs = curchisq
            curmodel = np.dot(coeffs, polys * evalTempl)
            models.append(curmodel)

        assert(np.isfinite(np.asscalar(curchisq)))
        chisq += np.asscalar(curchisq)

    chisq += 1e5 * outsides
    if getModel:
        ret = chisq, models
    else:
        ret = chisq
    return ret


def find_best(specdata, vel_grid, params_list, rot_params, resol_params, options=None,
              config=None):
    # find the best fit template and velocity from a grid
    cache = pylru.lrucache(100)
    chisq = np.zeros((len(vel_grid), len(params_list)))
    for j, curparam in enumerate(params_list):
        for i, v in enumerate(vel_grid):
            chisq[i, j] = get_chisq(specdata, v, curparam, rot_params,
                                    resol_params, options=options,
                                    config=config, cache=cache)
    xind = np.argmin(chisq)
    i1, i2 = np.unravel_index(xind, chisq.shape)
    probs = np.exp(-0.5 * (chisq[:, i2] - chisq[i1, i2]))
    besterr = (probs * vel_grid).sum() / probs.sum()
    return dict(bestchi=chisq[i1, i2],
                bestvel=vel_grid[i1],
                velerr=besterr,
                bestparam=params_list[i2],
                probs=probs)

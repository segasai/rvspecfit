import astropy.io.fits as pyfits
import functools
import scipy
import joblib
import numpy as np
import scipy.interpolate
import numpy.random
import functools
from scipy.constants.constants import speed_of_light
import gc
import os
import math
import spec_inter
import yaml
from tempfile import mkdtemp
from joblib import Memory
import frozendict
import random
import pylru


def freezeDict(d):
    if isinstance(d, dict):
        d1 = {}
        for k, v in d.items():
            d1[k] = freezeDict(v)

        return frozendict.frozendict(d1)
    else:
        return d


class SpecData:
    def __init__(self, name, lam, spec, espec):
        self.fd = {}
        self.fd['name'] = name
        self.fd['lam'] = lam
        self.fd['spec'] = spec
        self.fd['espec'] = espec
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
    
    # take into account the resolution
    if resol_params is not None:
        spec = convolve_resol(
            templ_spec, resol_params, rot_params, atm_params)

    # take into account the rotation of the star
    if rot_params is not None:
        spec = convolve_vrot(templ_spec, rot_params, atm_params)

    templ_tag = random.getrandbits(128)
    return outside, curInterp.lam, spec, templ_tag

def convolve_resol(*args):
    raise Exception("not implemented yet")
    return 1


def convolve_vrot(*args):
    """ convolve the spectrum with the stellar rotation velocity kernel
    """
    raise Exception("not implemented yet")
    return 1


def getRVInterpol(lam_templ, templ):
    """ Return the spectrum interpolator"""
    interpol = scipy.interpolate.UnivariateSpline(
        lam_templ, templ, s=0, k=3, ext=2)
    tag = random.getrandbits(128)
    return interpol, tag

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
        resol_params = tuple (resol_params)
    atm_params = tuple (atm_params)

    # iterate over multiple datasets
    for curdata in specdata:
        
        name = curdata.name

        outside, templ_lam, templ_spec, templ_tag = getCurTempl(
            name, atm_params, rot_params, 
            resol_params, config)
        
        #if the current point is outside the template grid 
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
            curtemplI, templ_tag = getRVInterpol(templ_lam, templ_spec)
            if cache is not None:
                cache[templ_tag] = curtemplI
        else:
            curtemplI, templ_tag = cache[templ_tag]

        evalTempl = evalRV(curtemplI, vel, curdata.lam)
        
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
    return vel_grid[i1], params_list[i2]

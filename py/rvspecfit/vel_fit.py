import sys
import time
import itertools
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

from rvspecfit import spec_fit
from rvspecfit import spec_inter

def firstguess(specdata, options=None,
               config=None,
               resolParams=None):
    min_vel = -1000
    max_vel = 1000
    vel_step0 = 5
    paramsgrid = {'logg': [1, 2, 3, 4, 5],
                  'teff': [3000, 5000, 8000, 10000],
                  'feh': [-2, -1, 0],
                  'alpha': [0]}
    vsinigrid = [None, 10, 100]
    specParams = spec_inter.getSpecParams(specdata[0].name, config)
    params = []
    for x in itertools.product(*paramsgrid.values()):
        curp = dict(zip(paramsgrid.keys(), x))
        curp = [curp[_] for _ in specParams]
        params.append(curp)
    vels_grid = np.arange(min_vel, max_vel, vel_step0)

    for vsini in vsinigrid:
        if vsini is None:
            rot_params = None
        else:
            rot_params = (vsini,)
        res = spec_fit.find_best(specdata, vels_grid, params,
                                 rot_params, resolParams,
                                 config=config, options=options)


def process(specdata, paramDict0, fixParam=None, options=None,
         config=None,
         resolParams=None):
    """
process(specdata, {'logg':10, 'teff':30, 'alpha':0, 'feh':-1,'vsini':0}, fixParam = ('feh','vsini'),
                config =config, resolParam = None)
    """

    # Configuration parameters, should be moved to the yaml file
    min_vel = -1000
    max_vel = 1000
    vel_step0 = 5  # the starting step in velocities
    vel_step = 1  # The final step in velocities TO BE CHECKED
    max_vsini = 500
    min_vsini = 1e-2
    min_vel_step = 0.2

    if config is None:
        raise Exception('Config must be provided')

    def mapVsini(vsini):
        return np.log(np.clip(vsini, min_vsini, max_vsini))

    def mapVsiniInv(x):
        return np.clip(np.exp(x), min_vsini, max_vsini)

    assert(np.allclose(mapVsiniInv(mapVsini(3)), 3))

    vels_grid = np.arange(min_vel, max_vel, vel_step0)
    curparam = spec_fit.param_dict_to_tuple(paramDict0, specdata[0].name,
                                            config=config)
    specParams = spec_inter.getSpecParams(specdata[0].name, config)
    if fixParam is None:
        fixParam = []

    if 'vsini' not in paramDict0:
        rot_params = None
        fitVsini = False
    else:
        rot_params = (paramDict0['vsini'],)
        if 'vsini' in fixParam:
            fitVsini = False
        else:
            fitVsini = True

    res = spec_fit.find_best(specdata, vels_grid, [curparam],
                             rot_params, resolParams,
                             config=config, options=options)
    best_vel = res['best_vel']

    def paramMapper(p0):
        # construct relevant objects for fitting from a numpy array vectors
        # taking into account which parameters are fixed
        ret = {}
        p0rev = list(p0)[::-1]
        ret['vel'] = p0rev.pop()
        if fitVsini:
            vsini = mapVsiniInv(p0rev.pop())
            ret['vsini'] = vsini
        else:
            if 'vsini' in fixParam:
                ret['vsini'] = paramDict0['vsini']
            else:
                ret['vsini'] = None
        if ret['vsini'] is not None:
            ret['rot_params'] = (ret['vsini'],)
        else:
            ret['rot_params'] = None
        ret['params'] = []
        for x in specParams:
            if x in fixParam:
                ret['params'].append(paramDict0[x])
            else:
                ret['params'].append(p0rev.pop())
        assert(len(p0rev) == 0)
        return ret

    startParam = [best_vel]

    if fitVsini:
        startParam.append(mapVsini(paramDict0['vsini']))

    for x in specParams:
        if x not in fixParam:
            startParam.append(paramDict0[x])

    def func(p):
        pdict = paramMapper(p)
        if pdict['vel'] > max_vel or pdict['vel']<min_vel:
            return 1e30
        chisq = spec_fit.get_chisq(specdata, pdict['vel'],
                                   pdict['params'], pdict['rot_params'],
                                   resolParams,
                                   options=options, config=config)
        return chisq
    method = 'Nelder-Mead'
    t1 = time.time()
    res = scipy.optimize.minimize(func, startParam, method=method,
                                  options={'fatol': 1e-3, 'xatol': 1e-2})
    best_param = paramMapper(res['x'])
    ret = {}
    ret['param'] = dict(zip(specParams, best_param['params']))
    if fitVsini:
        ret['vsini'] = best_param['vsini']
    ret['vel'] = best_param['vel']
    best_vel = best_param['vel']

    t2 = time.time()

    # For a given template measure the chi-square as a function of velocity to get the uncertaint

    # if the velocity is outside the range considered, something
    # is likely wrong with the object , so to prevent future failure
    # I just limit the velocity
    if best_vel > max_vel or best_vel < min_vel:
        print('Warning velocity too large...')
        if best_vel > max_vel:
            best_vel = max_vel
        else:
            best_vel = min_vel

    crit_ratio = 5  # we want the step size to be at least crit_ratio times smaller than the uncertainty

    # Here we are evaluating the chi-quares on the grid of
    # velocities to get the uncertainty
    while True:
        vels_grid = np.concatenate((np.arange(
            best_vel, min_vel, -vel_step)[::-1], np.arange(best_vel + vel_step, max_vel, vel_step)))
        res1 = spec_fit.find_best(specdata, vels_grid, [[ret['param'][_] for _ in specParams]],
                                  best_param['rot_params'], resolParams,
                                  config=config, options=options)
        if vel_step < res1['vel_err'] / crit_ratio or vel_step < min_vel_step:
            break
        else:
            vel_step = max(res1['vel_err'], vel_step) / crit_ratio * 0.8
            new_width = max(res1['vel_err'], vel_step) * 10
            min_vel = max(best_vel - new_width, min_vel)
            max_vel = min(best_vel + new_width, max_vel)
    t3 = time.time()
    ret['vel_err'] = res1['vel_err']
    outp = spec_fit.get_chisq(specdata, best_vel, [ret['param'][_] for _ in specParams],
                                     best_param['rot_params'],
                                     resolParams,
                                     options=options, config=config,
                                     full_output=True)
    ret['yfit'] = outp['models']
    ret['chisq'] = outp['chisq']
    ret['chisq_array'] = outp['chisq_array']
    return ret

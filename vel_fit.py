import astropy.io.fits as pyfits
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import spec_fit
import spec_inter
import scipy.optimize


def doit(specdata, paramDict0, fixParam=None, options=None,
         config=None,
         resolParams=None):
    """
doit(specdata, {'logg':10, 'teff':30, 'alpha':0, 'feh':-1,'vsini':0}, fixParam = ('feh','vsini'),
                config =config, resolParam = None)
"""
    maxnit = 5
    minvel = -700
    maxvel = 700
    velstep0 = 5
    maxRotVel = 500
    minRotVel = 1e-2

    if config is None:
        raise Exception('Config must be provided')

    def mapVsini(vsini):
        return np.log(np.clip(vsini, minRotVel, maxRotVel))

    def mapVsiniInv(x):
        return np.clip(np.exp(x), minRotVel, maxRotVel)

    assert(np.allclose(mapVsiniInv(mapVsini(3)), 3))

    normchiFeatureless = 0

    vels_grid = np.arange(minvel, maxvel, velstep0)
    curparam = spec_fit.param_dict_to_tuple(paramDict0, specdata[0].name,
                                            config=config)
    specParams = spec_inter.getSpecParams(specdata[0].name, config)

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
    bestvel = res['bestvel']

    def paramMapper(p0):
        # construct relevant objects for fitting from a numpy array vectors
        # taking into account which parameters are fixed
        ret = {}
        p0rev = list(p0)[::-1]
        ret['vel'] = p0rev.pop()
        if fitVsini:
            vsini = mapVsini(p0rev.pop())
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

    startParam = [bestvel]

    if fitVsini:
        starParam.append(mapVsiniInv(paramDict0['vsini']))
    else:
        for x in specParams:
            if x not in fixParam:
                startParam.append(paramDict0[x])
    print(startParam, fixParam, specParams)

    def func(p):
        pdict = paramMapper(p)
        chisq = spec_fit.get_chisq(specdata, pdict['vel'],
                                   pdict['params'], pdict['rot_params'],
                                   resolParams,
                                   options=options, config=config)
        print(pdict['params'], pdict['vel'], chisq)
        return chisq
    method = 'Nelder-Mead'
    res = scipy.optimize.minimize(func, startParam, method=method,
                                  options={'fatol': 1e-3, 'xatol': 1e-2})
    bestparam = paramMapper(res['x'])
    ret = {}
    ret['param'] = dict(zip(specParams, bestparam))
    if fitVsini:
        ret['vsini'] = bestparam['vsini']
    ret['vel'] = bestparam['vel']
    return ret

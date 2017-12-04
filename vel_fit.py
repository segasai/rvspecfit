import astropy.io.fits as pyfits
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import spec_fit
import spec_inter
import scipy.optimize
import itertools

def firstguess(specdata, options=None,
         config=None,
         resolParams=None):
    minvel=-1000
    maxvel=1000
    velstep0 = 5
    paramsgrid = {'logg':[1,2,3,4,5],
                  'teff':[3000,5000,8000,10000],
                  'feh':[-2,-1,0],
                  'alpha':[0]}
    vsinigrid = [None,10,100]
    specParams = spec_inter.getSpecParams(specdata[0].name, config)
    params = []
    for x in itertools.product(*paramsgrid.values()):
        curp = dict(zip(paramsgrid.keys(),x))
        curp = [curp[_] for _ in specParams]
        params.append(curp)
    vels_grid = np.arange(minvel, maxvel, velstep0)

    for vsini in vsinigrid:
        if vsini is None:
            rot_params  =None
        else:
            rot_params = (vsini,)
        res = spec_fit.find_best(specdata, vels_grid, params,
                             rot_params, resolParams,
                             config=config, options=options)


def doit(specdata, paramDict0, fixParam=None, options=None,
         config=None,
         resolParams=None):
    """
doit(specdata, {'logg':10, 'teff':30, 'alpha':0, 'feh':-1,'vsini':0}, fixParam = ('feh','vsini'),
                config =config, resolParam = None)
    """

    # Configuration parameters, should be moved to the yaml file
    minvel = -1000
    maxvel = 1000
    velstep0 = 5 # the starting step in velocities
    velstep = 1 # The final step in velocitie TO BE CHECKED
    maxRotVel = 500
    minRotVel = 1e-2
    minvelstep = 0.2 

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
    best_vel = res['best_vel']

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

    startParam = [best_vel]

    if fitVsini:
        startParam.append(mapVsiniInv(paramDict0['vsini']))

    for x in specParams:
        if x not in fixParam:
            startParam.append(paramDict0[x])

    def func(p):
        pdict = paramMapper(p)
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

    crit_ratio = 5 # we want the step size to be at least crit_ratio times smaller than the uncertainty
    while True:
        vels_grid = np.concatenate((np.arange (best_vel, minvel, velstep)[::-1], np.arange(best_vel+velstep, maxvel, velstep)))
        res1 = spec_fit.find_best(specdata, vels_grid, [[ret['param'][_] for _ in specParams]],
                             best_param['rot_params'], resolParams,
                             config=config, options=options)
        if velstep < res1['vel_err'] / crit_ratio or velstep<minvelstep:
            break
        else:
            velstep = max(res1['vel_err'],velstep)/crit_ratio*0.8
            newwidth = max(res1['vel_err'], velstep)*10 
            minvel = max(best_vel - newwidth, minvel)
            maxvel = min(best_vel + newwidth, maxvel)
    t3 = time.time()
    # print (t2-t1,t3-t2)
    ret['vel_err'] = res1['vel_err']
    chisq,yfit = spec_fit.get_chisq(specdata, best_vel
                               ,[ret['param'][_] for _ in specParams],
                               best_param['rot_params'],
                               resolParams,
                                    options=options, config=config,
                                    getModel=True)
    ret['yfit'] = yfit
    ret['chisq'] = chisq
    return ret

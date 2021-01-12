import time
import itertools
import numpy as np
import scipy.optimize
import numdifftools as ndf
from rvspecfit import spec_fit
from rvspecfit import spec_inter
import logging


def firstguess(specdata, options=None, config=None, resolParams=None):
    """ Compute the starting point parameter by just going over a
    small grid of templates

    Parameters
    ----------

    specdata: list of Specdata
        Spectroscopic dataset
    options: dict
        Optional dictionary of options
    config: dict
        Optional dictionary config
    resolParams: tuple
        Resultion parameters

    Returns
    -------
    bestpar: dict
        Dictionary of best parameters

    """
    min_vel = config['min_vel']
    max_vel = config['max_vel']
    vel_step0 = config['vel_step0']
    options = options or {}
    paramsgrid = {
        'logg': [1, 2, 3, 4, 5],
        'teff': [3000, 5000, 8000, 10000],
        'feh': [-2, -1, 0],
        'alpha': [0]
    }
    vsinigrid = [None, 10, 100]
    specParams = spec_inter.getSpecParams(specdata[0].name, config)
    params = []
    for x in itertools.product(*paramsgrid.values()):
        curp = dict(zip(paramsgrid.keys(), x))
        curp = [curp[_] for _ in specParams]
        params.append(curp)
    vels_grid = np.arange(min_vel, max_vel, vel_step0)
    best_chisq = np.inf
    for vsini in vsinigrid:
        if vsini is None:
            rot_params = None
        else:
            rot_params = (vsini, )
        res = spec_fit.find_best(specdata,
                                 vels_grid,
                                 params,
                                 rot_params,
                                 resolParams,
                                 config=config,
                                 options=options)
        if res['best_chi'] < best_chisq:
            bestpar = {}
            for i, k in enumerate(specParams):
                bestpar[k] = res['best_param'][i]
            if vsini is not None:
                bestpar['vsini'] = vsini
            best_chisq = res['best_chi']
    return bestpar


class VSiniMapper:
    def __init__(self, min_vsini, max_vsini):
        self.min_vsini = min_vsini
        self.max_vsini = max_vsini

    def forward(self, vsini):
        return np.log(np.clip(vsini, self.min_vsini, self.max_vsini))

    def inverse(self, x):
        return np.clip(np.exp(x), self.min_vsini, self.max_vsini)


class ParamMapper:
    """
    This class constructs the dictionary with human readable parameters
    out of the vector, as well as taking into accoutnt which params are fixed
"""
    def __init__(self,
                 specParams,
                 paramDict0,
                 fixParam,
                 vsiniMapper,
                 fitVsini=True):
        """ Initializes the class
        Parameters
        ----------
        specParams: list
            The list of names of spec parameters
        paramDict0: dict
            Dictionary with starting parameters (must include all the required
            model params)
        fixParam: list
            Which params to fix
        vsiniMapper: class
            Class that does the transformation from vsini to transformed
            clipped vsini
        fitVsini: bool
            Whether vsini is among fitted params or not
"""
        self.specParams = specParams
        self.paramDict0 = paramDict0
        self.fixParam = fixParam
        self.vsiniMapper = vsiniMapper
        self.fitVsini = fitVsini

    def forward(self, p0):
        """ Convert a vector into param dictionary """
        ret = {}
        p0rev = list(p0)[::-1]
        ret['vel'] = p0rev.pop()
        if self.fitVsini:
            vsini = self.vsiniMapper.inverse(p0rev.pop())
            ret['vsini'] = vsini
        else:
            if 'vsini' in self.fixParam:
                ret['vsini'] = self.paramDict0['vsini']
            else:
                ret['vsini'] = None
        if ret['vsini'] is not None:
            ret['rot_params'] = (ret['vsini'], )
        else:
            ret['rot_params'] = None
        ret['params'] = []
        for x in self.specParams:
            if x in self.fixParam:
                ret['params'].append(self.paramDict0[x])
            else:
                ret['params'].append(p0rev.pop())
        assert (len(p0rev) == 0)
        return ret


def chisq_func(p, args):
    """
The function computes the chi-square of the fit
This function is used for minimization
"""
    paramMapper = args['paramMapper']
    pdict = paramMapper.forward(p)
    if pdict['vel'] > args['max_vel'] or pdict['vel'] < args['min_vel']:
        return 1e30
    chisq = spec_fit.get_chisq(args['specdata'],
                               pdict['vel'],
                               pdict['params'],
                               pdict['rot_params'],
                               args['resolParams'],
                               options=args['options'],
                               config=args['config'])
    return chisq


def process(
    specdata,
    paramDict0,
    fixParam=None,
    options=None,
    config=None,
    resolParams=None,
):
    """ Process spectra by doing maximum likelihood fit to spectral data

    Parameters
    ----------

    specdata: list of SpecData
        List of spectroscopic datasets to fit
    paramDict0: dict
        Dictionary of parameters to start from
    fixParam: tuple
        Tuple of parameters that will be fixed
    options: dict
        Dictionary of options
    config: dict
        Configuration dictionary
    resolParams: tuple
        Tuple of parameters for the resolution of current spectrum.

    Returns
    -------
    ret: dict
        Dictionary of parameters
        Keys are 'yfit' -- the list of best-fit models
        'vel', 'vel_kurtosis', 'vel_skewness', 'vel_err' velocity and
        its constraints
        'param' -- parameter values
        'param_err' -- parameter uncertaintoes
        'chisq' -- -2*log(likelihood) value (does not equal to chisq, because
        of marginalization)
        'chisq_array' -- array of proper chi-squares of the mode

    Example
    -------

    >>> ret = process(specdata, {'logg':10, 'teff':30, 'alpha':0, 'feh':-1,
        'vsini':0}, fixParam = ('feh','vsini'),
                config=config, resolParams = None)

    """

    if config is None:
        raise Exception('Config must be provided')

    min_vel = config['min_vel']
    max_vel = config['max_vel']
    vel_step0 = config['vel_step0']  # the starting step in velocities
    max_vsini = config['max_vsini']
    min_vsini = config['min_vsini']
    min_vel_step = config['min_vel_step']
    second_minimizer = config['second_minimizer']
    options = options or {}

    vels_grid = np.arange(min_vel, max_vel, vel_step0)

    curparam = spec_fit.param_dict_to_tuple(paramDict0,
                                            specdata[0].name,
                                            config=config)
    specParams = spec_inter.getSpecParams(specdata[0].name, config)
    if fixParam is None:
        fixParam = []

    if 'vsini' not in paramDict0:
        rot_params = None
        fitVsini = False
    else:
        rot_params = (paramDict0['vsini'], )
        if 'vsini' in fixParam:
            fitVsini = False
        else:
            fitVsini = True
    t0 = time.time()

    # This takes the input template parameters and scans the velocity
    # grid with it
    res = spec_fit.find_best(specdata,
                             vels_grid, [curparam],
                             rot_params,
                             resolParams,
                             config=config,
                             options=options)
    best_vel = res['best_vel']

    # std_vec is the vector of standard deviations used to create
    # a simplex for Nelder mead
    startParam = [best_vel]
    std_vec = [5]

    vsiniMapper = VSiniMapper(min_vsini, max_vsini)

    if fitVsini:
        startParam.append(vsiniMapper.forward(paramDict0['vsini']))
        std_vec.append(0.1)

    for x in specParams:
        if x not in fixParam:
            startParam.append(paramDict0[x])
            std_vec.append({
                'logg': 0.5,
                'teff': 300,
                'feh': 0.5,
                'alpha': 0.25
            }.get(x) or 0.5)

    std_vec = np.array(std_vec)

    t1 = time.time()
    curval = np.array(startParam)
    R = np.random.RandomState(43434)
    curiter = 1
    maxiter = 2
    ndim = len(curval)
    simp = np.zeros((ndim + 1, ndim))
    minimize_success = True
    simp[0, :] = curval
    simp[1:, :] = (curval[None, :] +
                   np.array(std_vec)[None, :] * R.normal(size=(ndim, ndim)))
    paramMapper = ParamMapper(specParams,
                              paramDict0,
                              fixParam,
                              vsiniMapper,
                              fitVsini=fitVsini)
    args = dict(min_vel=min_vel,
                max_vel=max_vel,
                resolParams=resolParams,
                paramMapper=paramMapper,
                specdata=specdata,
                options=options,
                config=config)
    while True:
        res = scipy.optimize.minimize(chisq_func,
                                      curval,
                                      args=args,
                                      method='Nelder-Mead',
                                      options={
                                          'fatol': 1e-3,
                                          'xatol': 1e-2,
                                          'initial_simplex': simp,
                                          'maxiter': 10000,
                                          'maxfev': np.inf
                                      })
        curval = res['x']
        simp = res['final_simplex'][0]
        if res['success']:
            break
        if curiter == maxiter:
            logging.warning('Maximum number of iterations reached')
            minimize_success = False
            break
        curiter += 1

    t2 = time.time()
    if second_minimizer:
        res = scipy.optimize.minimize(chisq_func,
                                      res['x'],
                                      method='BFGS',
                                      args=args)
    t3 = time.time()
    best_param = paramMapper.forward(res['x'])
    ret = {}
    ret['param'] = dict(zip(specParams, best_param['params']))
    if fitVsini:
        ret['vsini'] = best_param['vsini']
    ret['vel'] = best_param['vel']
    best_vel = best_param['vel']

    # For a given template measure the chi-square as a function of velocity
    # to get the uncertainty

    # if the velocity is outside the range considered, something
    # is likely wrong with the object , so to prevent future failure
    # I just limit the velocity
    if best_vel > max_vel or best_vel < min_vel:
        logging.warning('Velocity too large...')
        if best_vel > max_vel:
            best_vel = max_vel
        else:
            best_vel = min_vel

    crit_ratio = 5  # we want the step size to be at least crit_ratio
    # times smaller than the uncertainty

    # Here we are evaluating the chi-quares on the grid of
    # velocities to get the uncertainty
    vel_step = vel_step0
    while True:
        vels_grid = np.concatenate(
            (np.arange(best_vel, min_vel, -vel_step)[::-1],
             np.arange(best_vel + vel_step, max_vel, vel_step)))
        res1 = spec_fit.find_best(specdata,
                                  vels_grid,
                                  [[ret['param'][_] for _ in specParams]],
                                  best_param['rot_params'],
                                  resolParams,
                                  config=config,
                                  options=options)
        best_vel = res1['best_vel']
        if vel_step < res1['vel_err'] / crit_ratio or vel_step < min_vel_step:
            break
        else:
            vel_step = max(res1['vel_err'], vel_step) / crit_ratio * 0.8
            new_width = max(res1['vel_err'], vel_step) * 10
            min_vel = max(best_vel - new_width, min_vel)
            max_vel = min(best_vel + new_width, max_vel)
    t4 = time.time()
    ret['vel'] = best_vel
    ret['vel_err'] = res1['vel_err']
    ret['vel_skewness'] = res1['skewness']
    ret['vel_kurtosis'] = res1['kurtosis']
    outp = spec_fit.get_chisq(specdata,
                              best_vel, [ret['param'][_] for _ in specParams],
                              best_param['rot_params'],
                              resolParams,
                              options=options,
                              config=config,
                              full_output=True)
    t5 = time.time()

    # compute the uncertainty of stellar params
    def hess_func(p):
        outp = spec_fit.get_chisq(specdata,
                                  best_vel,
                                  p,
                                  best_param['rot_params'],
                                  resolParams,
                                  options=options,
                                  config=config,
                                  full_output=True)
        return 0.5 * outp['chisq']

    # hess_step = np.maximum(
    #    1e-4 * np.abs(np.array([ret['param'][_] for _ in specParams])), 1e-4)
    hess_step = [{
        'vsini': 10,
        'logg': 1,
        'feh': 0.1,
        'alpha': 1,
        'teff': 100,
        'vrad': 10,
    }[_] for _ in specParams]
    hess_step = ndf.MinStepGenerator(base_step=hess_step, step_ratio=10)
    hessian = ndf.Hessian(
        hess_func, step=hess_step)([ret['param'][_] for _ in specParams])
    try:
        hessian_inv = scipy.linalg.inv(hessian)
        diag_hess = np.array(np.diag(hessian_inv))
        bad_diag_hess = diag_hess < 0
        diag_hess[bad_diag_hess] = 0
        diag_err = np.sqrt(diag_hess)
        diag_err[bad_diag_hess] = np.nan
    except np.linalg.LinAlgError:
        logging.warning('The inversion of the Hessian failed')
        diag_err = np.zeros(hessian.shape[0]) + np.nan
        #
    ret['param_err'] = dict(zip(specParams, diag_err))
    ret['minimize_success'] = minimize_success

    ret['yfit'] = outp['models']
    ret['chisq'] = outp['chisq']
    ret['chisq_array'] = outp['chisq_array']
    t6 = time.time()
    logging.debug('Timings process: %.4f %.4f %.4f %.4f, %.4f %.4f' %
                  (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5))
    return ret

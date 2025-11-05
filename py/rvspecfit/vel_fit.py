import time
import logging
import copy
import math
import itertools
import numpy as np
import scipy.optimize
import numdifftools as ndf
from rvspecfit import spec_fit
from rvspecfit import spec_inter


def firstguess(specdata,
               options=None,
               config=None,
               resolParams=None,
               vsinigrid=(None, 10, 100),
               paramsgrid=None):
    """ Compute the starting point template parameters and radial velocity
    by brute force looping over small grid of templates and a grid of
    radial velocities. This can be useful before starting a maximum-likelihood
    fit using vel_fit.process() or for initializing MCMC samples.

    Parameters
    ----------

    specdata: list of Specdata
        Spectroscopic dataset
    options: dict
        Optional dictionary of options
    config: dict
        Optional dictionary config
    resolParams: tuple
        Resolution parameters
    paramsgrid: dictionary
        (optional) dictionary of template parameters to iterate over
        The default value is

        .. code-block:: python

            paramsgrid = {
                'logg': [1, 2, 3, 4, 5],
                'teff': [3000, 5000, 8000, 10000],
                'feh': [-2, -1, 0],
                'alpha': [0]
            }
    vsinigrid: tuple
        (optional) list of vsinis to consider

    Returns
    -------
    bestpar: dict
        Dictionary of best parameters

    """
    min_vel = config['min_vel']
    max_vel = config['max_vel']
    vel_step0 = config['vel_step0']
    options = options or {}
    if paramsgrid is None:
        paramsgrid = {
            'logg': [1, 2, 3, 4, 5],
            'teff': [3000, 5000, 8000, 10000],
            'feh': [-2, -1, 0],
            'alpha': [0]
        }
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
                                 rot_params=rot_params,
                                 resol_params=resolParams,
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
        self.log_min_vsini = np.log(min_vsini)
        self.log_max_vsini = np.log(max_vsini)
        self.min_vsini = min_vsini
        self.max_vsini = max_vsini

    def forward(self, vsini):
        """ Convert human normal vsini into
        log-transformed clipped vsini """
        return np.log(np.clip(vsini, self.min_vsini, self.max_vsini))

    def inverse(self, x):
        """ Undo the transformation.
        Return proper vsini
        """
        return np.exp(np.clip(x, self.log_min_vsini, self.log_max_vsini))


class ParamMapper:
    """
    This class constructs the dictionary with human readable parameters
    out of the vector, as well as taking into account which params are fixed

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
        """ Convert a parameter vector into param dictionary
        Argument order in the vector
        velocity [vsini] stellar parameters
        vsini is optional
        stellar parameters are in the order given by specParams
        and the ones with fixParam are excluded
        
        Parameters
        ----------
        p0: array
            Vector with parameter values

        Returns
        -------
        ret: dictionary
            The dict with all the parameter values
        """
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

    def get_fitted_params(self):
        ret = ['vel']
        if self.fitVsini:
            ret.append('vsini')
        for x in self.specParams:
            if x not in self.fixParam:
                ret.append(x)
        return ret


def chisq_func0(pdict, args, outside_penalty=True):
    # this is the generic function that returns
    # chi-square + prior
    # it is called by the
    chisq = 0
    if args.get('priors') is not None:
        priors = args['priors']
        for i, k in enumerate(args['paramMapper'].specParams):
            if k in priors:
                chisq += ((priors[k][0] - pdict['params'][i]) /
                          priors[k][1])**2

    chisq += spec_fit.get_chisq(args['specdata'],
                                pdict['vel'],
                                pdict['params'],
                                pdict['rot_params'],
                                args['resolParams'],
                                options=args['options'],
                                config=args['config'],
                                outside_penalty=outside_penalty)
    return chisq


def chisq_func(p, args):
    """
    The function computes the chi-square of the fit
    This function is used for minimization

    Parameters
    ----------
    p: array_like
        Vector of parameters to be fitted (velocity, vsini, stellar params)
    args: dict
        Dictionary containing specdata, config, and other fitting arguments

    Returns
    -------
    chisq: float
        Chi-square value for the given parameters
    """
    paramMapper = args['paramMapper']
    pdict = paramMapper.forward(p)
    if (pdict['vel'] > args['max_vel'] or pdict['vel'] < args['min_vel']
            or (~np.isfinite(pdict['params'])).any()):
        return 1e30
    ret = chisq_func0(pdict, args)
    return ret


def hess_func(p, pdict, args):
    """
The function computes the 0.5*chi-square
and takes as input vector of parameters (not transformed)
pdict is for fixedparameters
specParams is the list of names of parameters that we are varying
"""
    pdict['params'][:] = p[:]
    ret = 0.5 * chisq_func0(pdict, args)  # , outside_penalty=False)
    return ret


def _get_simplex_start(best_vel,
                       fixParam=None,
                       specParamNames=None,
                       paramDict0=None,
                       vsiniMapper=None,
                       fitVsini=None):
    """
    Create starting simplex and starting point for optimization
    This is a deterministic simplex
    """
    # std_vec is the vector of standard deviations used to create
    # a simplex for Nelder mead
    startParam = [best_vel]
    std_vec = [5]

    # second parameter is vsini
    if fitVsini:
        startParam.append(vsiniMapper.forward(paramDict0['vsini']))
        std_vec.append(0.1)

    for x in specParamNames:
        if x not in fixParam:
            startParam.append(paramDict0[x])
            std_vec.append({
                'logg': 0.5,
                'teff': 300,
                'feh': 0.5,
                'alpha': 0.25
            }.get(x) or 0.5)

    curval = np.array(startParam)
    std_vec = np.array(std_vec)

    ndim = len(curval)
    R = np.random.RandomState(43434)
    simp = np.zeros((ndim + 1, ndim))
    # first point is a starting point
    simp[0, :] = curval
    simp[1:, :] = (curval[None, :] +
                   np.array(std_vec)[None, :] * R.normal(size=(ndim, ndim)))
    return curval, simp


def _find_best_vel_iterate(best_vel,
                           min_vel,
                           max_vel,
                           vel_step0,
                           specdata=None,
                           best_param=None,
                           resolParams=None,
                           config=None,
                           options=None,
                           min_vel_step=None):
    """
    This function perform iterations around the current best_vel
    to make sure the velocity posterior is well sampled.
    It returns the best velocity, the uncertainty, skewness and kurtosis
    """
    # if the velocity is outside the range considered, something
    # is likely wrong with the object , so to prevent future failure
    # I just limit the velocity
    if best_vel > max_vel or best_vel < min_vel:
        logging.warning('Velocity too large...')
        if best_vel > max_vel:
            best_vel = max_vel
        else:
            best_vel = min_vel

    def func(vels_grid):
        res1 = spec_fit.find_best(specdata,
                                  vels_grid, [best_param['params']],
                                  rot_params=best_param['rot_params'],
                                  resol_params=resolParams,
                                  config=config,
                                  options=options)
        best_vel = res1['best_vel']
        cur_err = res1['vel_err']
        return best_vel, cur_err, res1

    best_vel, best_err, res1 = _minimum_sampler(func, best_vel, min_vel,
                                                max_vel, vel_step0,
                                                min_vel_step)

    return best_vel, best_err, res1['skewness'], res1['kurtosis']


def _minimum_sampler(func,
                     best_vel,
                     min_vel,
                     max_vel,
                     vel_step0,
                     min_vel_step,
                     crit_ratio=5,
                     goal_width=10):
    """
    This function tries to find the minimum value and the error.
    The key point is that it tries to ensure the step on the grid
    is small enough. Also crucially we start from broad range of velocities
    to ensure we can capture multiple peaks in the CCF.

    Parameters:
    -----------
    func: function
        This function should return the tuple where the first two items
        are best value, uncertainty.
    best_vel: float
        Initial best guess
    min_vel: float
        Lower boundary to consider
    max_vel: float
        Upper boundary to consider
    vel_step0: float
        Starting step size
    min_vel_step: float
        Stop if the step size is below the value
    crit_ratio: float
        Require that the uncertainty/step size is bigger than that
    goal_width: float
        make sure the grid is at leas that many sigma wide
    """
    # we want the step size to be at least crit_ratio
    # times smaller than the uncertainty

    # Here we are evaluating the chi-quares on the grid of
    # velocities to get the uncertainty
    vel_step = vel_step0
    for it in range(10):
        # at each iteration I update two things
        # step size and min,max values of the velocity window

        # velocity grid that goes from min_vel to max_vel and goes
        # exactly through best_vel
        vels_grid = np.arange(
            math.ceil((min_vel - best_vel) / vel_step) * vel_step,
            max_vel - best_vel, vel_step) + best_vel
        best_vel, cur_err, res1 = func(vels_grid)
        # I stop if the step becomes smaller than the some fraction of the
        # velocity error or if step is just too small
        if vel_step < cur_err / crit_ratio or vel_step < min_vel_step:
            break
        else:
            # construct new velocity step and width
            # When choosing the width there are two things to consider
            # At first the error may be incorrect if the step is too large
            # so I need be careful that the width is not based on that.
            if vel_step > cur_err:
                # we are not resolving the uncertainty properly so the
                # uncertainty is essentially not correct
                # so I essentially use vel_step as indicator of the velocity
                vel_step_new = vel_step / crit_ratio
                width_new = vel_step * goal_width
            else:
                # vel_step < cur_err
                # normally resolved regime
                vel_step_new = cur_err / crit_ratio * 0.8
                # 0.8 is there to ensure that I satisfy the
                # err/step > crit_ratio from first iteration
                width_new = cur_err * goal_width
            # It is guaranteed that the new velocity step is smaller than
            # before

            min_vel = max(best_vel - width_new, min_vel)
            max_vel = min(best_vel + width_new, max_vel)
            vel_step = vel_step_new
    if it > 5:
        logging.warning(
            'More than 5 iterations we used in finding the velocity error')
    return best_vel, cur_err, res1


def get_hess_inv(param_names):
    """
    The inverse hessian is approximately the errors^2
    Here we set it up.
    """

    default_err0 = 0.1
    teff_err0 = 50
    rv_err0 = 1
    diag = np.zeros(len(param_names)) + default_err0**2
    teff_idx = np.nonzero(np.asarray(param_names) == 'teff')[0][0]
    diag[teff_idx] = teff_err0**2
    diag[0] = rv_err0**2
    hess_inv0 = np.diag(diag)
    return hess_inv0


def _uncertainties_from_hessian(hessian):
    """
    Take the hessian and return the uncertainties vector and
    covariance matrix
    Here I also protect all sorts of failures when encountering
    bad hessian
    """
    diag_hessian = np.diag(hessian)
    inv_diag_hessian = 1. / (diag_hessian + (diag_hessian == 0))
    inv_diag_hessian[diag_hessian == 0] = np.inf
    bad_hessian = False
    try:
        hessian_inv = scipy.linalg.inv(hessian)
    except (np.linalg.LinAlgError, ValueError):
        bad_hessian = True
        logging.warning('The inversion of the Hessian failed')
        # trying to invert the diagonal
        hessian_inv = np.diag(inv_diag_hessian)

    # this the default one
    diag_err0 = np.array(np.diag(hessian_inv))

    # this is just inverting the diagonal
    diag_err1 = inv_diag_hessian
    bad_err0 = diag_err0 < 0
    bad_err1 = diag_err1 < 0
    if bad_err0.any():
        bad_hessian = True
    sub1 = bad_err0 & (~bad_err1)
    sub2 = bad_err0 & bad_err1

    diag_err0[sub1] = diag_err1[sub1]
    diag_err0[sub2] = 0
    diag_err = np.sqrt(diag_err0)
    diag_err[sub2] = np.nan
    if (~np.isfinite(diag_err)).sum() != 0:
        bad_par = (np.nonzero(~np.isfinite(diag_err))[0]).tolist()
        bad_hessian = True
        logging.debug(f'not finite uncertainty for params {bad_par}')
    return diag_err, hessian_inv, bad_hessian


def process(specdata,
            paramDict0,
            fixParam=None,
            options=None,
            config=None,
            resolParams=None,
            priors=None):
    """
    Process spectra by doing maximum likelihood fit to spectral data

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
    priors: dict (optional)
        Extra dictionary with Normal priors on paramaters
        I.e. {'teff':(5000,10)} for N(5000,10) prior on the
        effective temperature
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

    Examples
    --------

    >>> ret = process(specdata, {'logg':10, 'teff':30, 'alpha':0, 'feh':-1,
        'vsini':0}, fixParam = ('feh','vsini'),
                config=config, resolParams = None)

    """

    if config is None:
        raise RuntimeError('Config must be provided')
    if isinstance(specdata, spec_fit.SpecData):
        specdata = [specdata]

    min_vel = config['min_vel']
    max_vel = config['max_vel']
    vel_step0 = config['vel_step0']  # the starting step in velocities
    max_vsini = config['max_vsini']
    min_vsini = config['min_vsini']
    min_vel_step = config['min_vel_step']
    second_minimizer = config.get('second_minimizer') or False
    options = options or {}

    vels_grid = np.arange(min_vel, max_vel, vel_step0)

    curparam = spec_fit.param_dict_to_tuple(paramDict0,
                                            specdata[0].name,
                                            config=config)
    specParamNames = spec_inter.getSpecParams(specdata[0].name, config)
    if fixParam is None:
        fixParam = []

    vsiniMapper = None
    if 'vsini' not in paramDict0:
        rot_params = None
        fitVsini = False
    else:
        rot_params = (paramDict0['vsini'], )
        if 'vsini' in fixParam:
            fitVsini = False
        else:
            fitVsini = True
            vsiniMapper = VSiniMapper(min_vsini, max_vsini)

    t0 = time.time()

    # This takes the input template parameters and scans the velocity
    # grid with it
    res = spec_fit.find_best(specdata,
                             vels_grid, [curparam],
                             rot_params=rot_params,
                             resol_params=resolParams,
                             config=config,
                             options=options)
    best_vel = res['best_vel']

    t1 = time.time()

    curval, simplex = _get_simplex_start(best_vel,
                                         fixParam=fixParam,
                                         specParamNames=specParamNames,
                                         paramDict0=paramDict0,
                                         vsiniMapper=vsiniMapper,
                                         fitVsini=fitVsini)

    paramMapper = ParamMapper(specParamNames,
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
                config=config,
                priors=priors)
    minimize_success = True
    curiter = 1
    maxiter = 2
    hess_inv0 = get_hess_inv(paramMapper.get_fitted_params())
    while True:
        res0 = scipy.optimize.minimize(chisq_func,
                                       curval,
                                       args=args,
                                       method='Nelder-Mead',
                                       options={
                                           'fatol': 1e-3,
                                           'xatol': 1e-2,
                                           'initial_simplex': simplex,
                                           'maxiter': 10000,
                                           'maxfev': np.inf
                                       })
        curval = res0['x']
        simplex = res0['final_simplex'][0]
        if res0['success']:
            break
        if curiter == maxiter:
            logging.warning('Maximum number of iterations reached')
            minimize_success = False
            break
        curiter += 1

    t2 = time.time()
    if second_minimizer:
        res = scipy.optimize.minimize(chisq_func,
                                      res0['x'],
                                      method='BFGS',
                                      args=args,
                                      options=dict(hess_inv0=hess_inv0))
    else:
        res = res0
    t3 = time.time()
    best_param = paramMapper.forward(res['x'])
    ret = {}
    ret['param'] = dict(zip(specParamNames, best_param['params']))
    if fitVsini:
        ret['vsini'] = best_param['vsini']
    ret['vel'] = best_param['vel']
    best_vel = best_param['vel']

    # For a given template measure the chi-square as a function of velocity
    # to get the uncertainty
    best_vel, vel_err, vel_skewness, vel_kurtosis = _find_best_vel_iterate(
        best_vel,
        min_vel,
        max_vel,
        vel_step0,
        specdata=specdata,
        best_param=best_param,
        resolParams=resolParams,
        config=config,
        options=options,
        min_vel_step=min_vel_step)

    t4 = time.time()
    ret['vel'] = best_vel
    ret['vel_err'] = vel_err
    ret['vel_skewness'] = vel_skewness
    ret['vel_kurtosis'] = vel_kurtosis
    outp = spec_fit.get_chisq(specdata,
                              best_vel,
                              best_param['params'],
                              best_param['rot_params'],
                              resolParams,
                              options=options,
                              config=config,
                              full_output=True)
    t5 = time.time()

    # compute the uncertainty of stellar params
    best_param_TMP = copy.deepcopy(best_param)

    def hess_func_wrap(p):
        return hess_func(p, best_param_TMP, args)

    hess_step = [{
        'vsini': 1 / 100,
        'logg': 0.1 / 100,
        'feh': 0.1 / 100,
        'alpha': .01 / 100,
        'teff': 1 / 100,
        'vrad': 1 / 100,
    }[_] for _ in specParamNames]
    hess_step_gen = ndf.MinStepGenerator(base_step=hess_step)
    for i in range(2):
        # perform two iterations if there is an issue
        hessian = ndf.Hessian(hess_func_wrap, step=hess_step_gen)(
            [ret['param'][_] for _ in specParamNames])
        diag_err, covar_mat, bad_hessian = _uncertainties_from_hessian(hessian)
        if bad_hessian:
            hess_step_gen = None
            logging.warning(
                'Performing two iterations of hessian determination')

    ret['param_err'] = dict(zip(specParamNames, diag_err))
    ret['param_covar'] = covar_mat
    ret['minimize_success'] = minimize_success
    ret['bad_hessian'] = bad_hessian
    ret['yfit'] = outp['models']
    ret['raw_models'] = outp['raw_models']
    ret['chisq'] = outp['chisq']
    ret['logl'] = outp['logl']
    ret['chisq_array'] = outp['chisq_array']
    ret['npix_array'] = outp['npix_array']
    t6 = time.time()
    logging.debug('Timings process: %.4f %.4f %.4f %.4f, %.4f %.4f' %
                  (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5))
    return ret

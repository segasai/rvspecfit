import numpy as np
import numpy.random
import scipy.interpolate
import os
import subprocess
from random import random, seed
import pickle
import dill
import argparse
import scipy.stats
import spec_fit


class CCFConfig:
    def __init__(self, logl0=None, logl1=None, npoints=None, splinestep=1000,
                 maxcontpts=30):
        # slinestep is the smoothing step for the continuum in km/s
        # but the number of points in the continuum fit will be below  maxcontpts
        self.logl0 = logl0
        self.logl1 = logl1
        self.npoints = npoints
        self.maxcontpts = maxcontpts
        self.splinestep = max(
            splinestep, 3e5 * (np.exp((logl1 - logl0) / self.maxcontpts)-1))


def get_revision():
    """ get the git revision of the code"""
    try:
        fname = os.path.dirname(os.path.realpath(__file__))
        tmpout = subprocess.Popen(
            'cd ' + fname + ' ; git log -n 1 --pretty=format:%H -- make_nd.py',
            shell=True, bufsize=80, stdout=subprocess.PIPE).stdout
        revision = tmpout.read()
        return revision
    except:
        return ''


git_rev = get_revision()


def get_cont(lam0, spec0, espec0, ccfconf=None, frac=10):
    # get the continuum in bins of spectra

    lammin = lam0.min()
    N = np.log(lam0.max() / lammin)/np.log(1+ccfconf.splinestep/3e5)
    N = int(np.ceil(N))
    nodes = lammin * np.exp(np.arange(N) * np.log(1+ ccfconf.splinestep / 3e5))
    nodesedges = lammin * np.exp((-0.5 + np.arange(N + 1))
                           * np.log(1+ccfconf.splinestep / 3e5))
    medspec = np.median(spec0)
    BS = scipy.stats.binned_statistic(lam0, spec0, 'median', bins=nodesedges)
    p0 = np.log(BS.statistic)
    p0[~np.isfinite(p0)] = np.log(medspec)

    lam, spec, espec = [x[::frac] for x in [lam0, spec0, espec0]]

    res = scipy.optimize.minimize(res_fit, p0,
                                  args=(spec, espec, nodes, lam),
                                  jac=False,
                                  # method='Nelder-Mead'
                                  method='BFGS'
                                  )['x']
    cont = res_fit(res, spec0, espec0, nodes, lam0, getModel=True)
    #from idlplot import plot,oplot
    #import matplotlib.pyplot as plt
    #plot (lam0, spec0,xlog=True)
    #oplot(lam0, cont,color='red')
    #1/0
    #plt.draw()
    #plt.pause(0.1)

    return cont


def res_fit(p, spec=None, espec=None, nodes=None, lam=None, getModel=False):
    II = scipy.interpolate.UnivariateSpline(nodes, p, s=0)
    model = np.exp(II(lam))
    if getModel:
        return model
    res = (spec - model) / espec
    val = np.abs(res).sum()
    # I may need to mask outliers here...
    if False:
        from idlplotInd import plot,oplot
        plot(spec)
        oplot(model,color='red')

    if not np.isfinite(val):
        return 1e30
    return val


def apodize(y):
    frac = 0.15
    l = len(y)
    x = np.arange(l) * 1.
    mask = 1 + 0 * x
    ind = x < (frac * l)
    mask[ind] = (1 - np.cos(x[ind] / frac / l * np.pi)) * .5
    ind = (x - l + 1) > (-frac * l)
    mask[ind] = (1 - np.cos((l - x - 1)[ind] / frac / l * np.pi)) * .5
    return mask * y


def pad(x, y):
    l = len(y)
    l1 = int(2**np.ceil(np.log(l) / np.log(2)))
    delta1 = int((l1 - l) / 2)
    delta2 = int((l1 - l) - delta1)
    y2 = np.concatenate((np.zeros(delta1), y, np.zeros(delta2)))
    deltax = x[1] - x[0]
    x2 = np.concatenate((np.arange(-delta1, 0) * deltax + x[0], x,
                         x[-1] + deltax * (1 + np.arange(delta2))))
    return x2, y2


def preprocess_models_doer(logl, lammodels, m0, vel, ccfconf=None):
    if vel is not None and vel!=0:
        m = spec_fit.convolve_vsini(lammodels, m0, vel)
    else:
        m = m0
    cont = get_cont(lammodels, m, m * 0 + 1e-5, ccfconf=ccfconf)

    c_model = scipy.interpolate.interp1d(np.log(lammodels), m / cont)(logl)
    c_model = c_model - np.mean(c_model)
    ca_model = apodize(c_model)
    xlogl, cpa_model = pad(logl, ca_model)
    std = (cpa_model**2).sum()**.5
    cpa_model /= std
    return xlogl, cpa_model, std


def preprocess_data(lam, spec0, espec, ccfconf=None, badmask=None):
    logl = np.linspace(ccfconf.logl0, ccfconf.logl1, ccfconf.npoints)
    curespec = espec.copy()
    curspec = spec0.copy()
    curespec[~badmask] = curespec[~badmask] * 0 + 1e9
    cont = get_cont(lam, curspec, curespec, ccfconf=ccfconf)
    c_spec = spec0 / cont
    c_spec = c_spec - np.median(c_spec)
    ca_spec = apodize(c_spec)
    if badmask is not None:
        ca_spec[badmask] = 0
    ca_spec = scipy.interpolate.interp1d(np.log(lam), ca_spec,bounds_error=False,
                                        fill_value=0, kind='linear')(logl)
    lam1, cap_spec = pad(logl, ca_spec)
    #1/0
    return cap_spec


def preprocess_models(lammodels, models, params, ccfconf, vsinis=None):
    logl = np.linspace(ccfconf.logl0, ccfconf.logl1, ccfconf.npoints)
    res = []
    retparams = []
    norms = []
    if vsinis is None:
        vsinis = [None]
    resA = []
    vsinisList = []
    for imodel, m0 in enumerate(models):
        print(imodel, len(models))
        for vel in vsinis:
            retparams.append(params[imodel])
            xlogl, cpa_model, std = preprocess_models_doer(
                logl, lammodels, m0, vel, ccfconf=ccfconf)
            vsinisList.append(vel)
            norms.append(std)
            res.append(cpa_model)
    return xlogl, res, retparams, vsinisList, norms

def dosetup(HR, ccfconf, prefix=None, oprefix=None, every=10, vsinis=None):
    "Prepare the N-d interpolation objects "

    postf = ''
    with open('%s/specs_%s%s.pkl' % (prefix, HR, postf), 'rb') as fp:
        D = pickle.load(fp)
        vec, specs, lam, parnames = D['vec'], D['specs'], D['lam'], D['parnames']
        del D

    ndim = len(vec[:, 0])

    specs = specs[::every, :]
    vec = vec.T[::every, :]
    nspec, lenspec = specs.shape

    xlogl, models, params, vsinis, norms = preprocess_models(
        lam, np.exp(specs), vec, ccfconf, vsinis=vsinis)
    ffts = np.array([np.fft.fft(x) for x in models])
    savefile = '%s/ccf_%s%s.pkl' % (oprefix, HR, postf)
    dHash = {}
    dHash['params'] = params
    dHash['ffts'] = np.array(ffts)
    dHash['models'] = np.array(models)
    dHash['ccfconf'] = ccfconf
    dHash['loglambda'] = xlogl
    dHash['vsinis'] = vsinis
    dHash['parnames'] = parnames

    with open(savefile, 'wb') as fp:
        pickle.dump(dHash, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--oprefix', type=str)
    parser.add_argument('--setup', type=str)
    parser.add_argument('--lambda0', type=float)
    parser.add_argument('--lambda1', type=float)
    parser.add_argument('--step', type=float)
    parser.add_argument('--vsinis', type=str, default=None)
    parser.add_argument('--every', type=int, default=30)

    args = parser.parse_args()

    npoints = (args.lambda1 - args.lambda0) / args.step
    import make_ccf
    ccfconf = make_ccf.CCFConfig(logl0=np.log(args.lambda0),
                        logl1=np.log(args.lambda1),
                        npoints=npoints)

    if args.vsinis is not None:
        vsinis = [float(_) for _ in args.vsinis.split(',')]
    else:
        vsinis = None
    dosetup(args.setup, ccfconf, args.prefix, args.oprefix, args.every, vsinis)

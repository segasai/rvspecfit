import pickle
import argparse
import numpy as np
import scipy.interpolate
import scipy.stats
import spec_fit
import make_ccf
import utils

git_rev = utils.get_revision()

class CCFConfig:
    """ Configuration class for cross-correlation functions """

    def __init__(self, logl0=None, logl1=None, npoints=None, splinestep=1000,
                 maxcontpts=20):
        """
        Configure the cross-correlation

        Parameters
        ----------
        logl0: float
             The natural logarithm of the wavelength of the beginning of the CCF
        logl1: float
             The natural logarithm of the wavelength of the end of the CCF
        npoints: integer
             The number of points in the cross correlation functions
        splinestep: float, optional
             The stepsize in km/s that determine the smoothness of the continuum
             fit
        maxcontpts: integer, optional
             The maximum number of points used for the continuum fit
        """

        self.logl0 = logl0
        self.logl1 = logl1
        self.npoints = npoints
        self.maxcontpts = maxcontpts
        self.splinestep = max(
            splinestep, 3e5 * (np.exp((logl1 - logl0) / self.maxcontpts) - 1))

def get_continuum(lam0, spec0, espec0, ccfconf=None, bin=11):
    """
    Determine the continuum of the spectrum by fitting a spline

    Parameters:
    ----------

    lam0: numpy array
        The wavelength vector
    spec0: numpy array
        The spectral vector
    espec0: numpy array
        The vector of spectral uncertainties
    ccfconf: CCFConfig object
        The CCF configuration object
    bin: integer, optional
        The input spectrum will be binned by median filter by this number before
        the fit

    Returns:
    --------
    cont: numpy array
        The continuum vector
    """

    lammin = lam0.min()
    N = np.log(lam0.max() / lammin) / np.log(1 + ccfconf.splinestep / 3e5)
    N = int(np.ceil(N))
    # Determine the nodes of the spline used for the continuum fit
    nodes = lammin * np.exp(np.arange(N) *
                            np.log(1 + ccfconf.splinestep / 3e5))
    nodesedges = lammin * np.exp((-0.5 + np.arange(N + 1))
                                 * np.log(1 + ccfconf.splinestep / 3e5))
    medspec = np.median(spec0)
    BS = scipy.stats.binned_statistic(lam0, spec0, 'median', bins=nodesedges)
    p0 = np.log(BS.statistic)
    p0[~np.isfinite(p0)] = np.log(medspec)

    lam, spec, espec = (lam0[::bin], scipy.signal.medfilt(spec0, bin)[::bin],
                        scipy.signal.medfilt(espec0, bin)[::bin])

    res = scipy.optimize.minimize(fit_loss, p0,
                                  args=(spec, espec, nodes, lam),
                                  jac=False,
                                  # method='Nelder-Mead'
                                  method='BFGS'
                                  )['x']
    cont = fit_loss(res, spec0, espec0, nodes, lam0, getModel=True)
    return cont


def fit_loss(p, spec=None, espec=None, nodes=None, lam=None, getModel=False):
    """
    Return the loss function (L1 norm) of the continuum fit_loss

    Parameters:
    p: numpy array
        Array with fit parameters
    spec: numpy array
        Spectrum that is being fitted
    espec: numpy array
        Error vector
    nodes: numpy array
        The location of the nodes of the spline fit
    lam: numpy array
        The wavelength vector
    getModel: boolean, optional
        If true return the bestfit model instead of the loss function

    Returns:
    loss: real
        The loss function of the fit
    model: numpy array (optional, depending on getModel parameter)
        The evaluated model
    """
    II = scipy.interpolate.UnivariateSpline(nodes, p, s=0,k=2)
    model = np.exp(II(lam))
    if getModel:
        return model
    res = (spec - model) / espec

    val = np.abs(res).sum()
    if False:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(lam,spec,'k')
        plt.plot(lam,model,'r')
        plt.draw()
        plt.pause(0.01)
        print (val)
    # I may need to mask outliers here...
    if not np.isfinite(val):
        return 1e30
    return val


def apodize(spec):
    """
    Apodize the spectrum

    Parameters:
    -----------
    spec: numpy array
        The input numpy array
    """
    frac = 0.15
    l = len(spec)
    x = np.arange(l) * 1.
    mask = 1 + 0 * x
    ind = x < (frac * l)
    mask[ind] = (1 - np.cos(x[ind] / frac / l * np.pi)) * .5
    ind = (x - l + 1) > (-frac * l)
    mask[ind] = (1 - np.cos((l - x - 1)[ind] / frac / l * np.pi)) * .5
    return mask * spec


def pad(x, y):
    """
    Pad the input array to the power of two lengths

    Parameters:
    -----------
    x: numpy array
        wavelength vector
    y: numpy array
        spectrum

    Returns:
    --------
    x2: numpy array
        New wavelength vector
    y2: numpy array
        New spectrum vector
    """
    l = len(y)
    l1 = int(2**np.ceil(np.log(l) / np.log(2)))
    delta1 = int((l1 - l) / 2)  # extra pixels on the left
    delta2 = int((l1 - l) - delta1)  # extra pixels on the right
    y2 = np.concatenate((np.zeros(delta1), y, np.zeros(delta2)))
    deltax = x[1] - x[0]
    if np.allclose(np.diff(x), deltax):
        x2 = np.concatenate((np.arange(-delta1, 0) * deltax + x[0], x,
                             x[-1] + deltax * (1 + np.arange(delta2))))
    elif np.allclose(np.diff(np.log(x)), np.log(x[1] / x[0])):
        ratx = x[1] / x[0]
        x2 = np.concatenate(deltax**(np.arange(-delta1, 0) * x[0], x,
                                     x[-1] * deltax ** (1 + np.arange(delta2))))
    else:
        raise Exception(
            'the wavelength axis is neither logarithmic, nor linear')
    return x2, y2


def preprocess_model(logl, lammodel, model0, vsini=None, ccfconf=None):
    """
    Take the input template model and return prepared for FFT vectors.
    That includes padding, apodizing and normalizing by continuum

    Parameters:
    -----------
    logl: numpy array
        The array of log wavelengths on which we want to outputted spectra
    lammodel: numpy array
        The wavelength array of the model
    model0: numpy array
        The initial model spectrum
    vsini: float, optional
        The stellar rotation, vsini
    ccfconf: CCFConfig object, required
        The CCF configuration object

    Returns:
    --------

    xlogl: Numpy array
        The log-wavelength of the resulting processed model0
    cpa_model: Numpy array
        The continuum normalized/subtracted, apodized and padded spectrum
    """
    if vsini is not None and vsini != 0:
        m = spec_fit.convolve_vsini(lammodel, model0, vsini)
    else:
        m = model0
    cont = get_continuum(lammodel, m, np.maximum(m*1e-5,1e-2*np.median(m)), ccfconf=ccfconf)

    c_model = scipy.interpolate.interp1d(np.log(lammodel), m / cont)(logl)
    c_model = c_model - np.mean(c_model)
    ca_model = apodize(c_model)
    xlogl, cpa_model = pad(logl, ca_model)
    std = (cpa_model**2).sum()**.5
    if std>1e5:
        print ('WARNING something went wrong with the spectrum ormalization')
    cpa_model /= std
    return xlogl, cpa_model


def preprocess_model_list(lammodels, models, params, ccfconf, vsinis=None):
    """
    Apply preprocessing to the array of models

    Parameters:
    lammodels: numpy array
        The array of wavelengths of the models (assumed to be the same for all models)
    models: numpy array
        The 2D array of modell with the shape [number_of_models, len_of_model]
    params: numpy array
        The 2D array of template parameters (i.e. stellar atmospheric parameters
        ) with the shape [number_of_models,length_of_parameter_vector]
    ccfconf: CCFConfig object
        CCF configuration
    vsinis: list of floats
        The list of possible Vsini values to convolve model spectra with
        Could be None
    """

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
            xlogl, cpa_model = preprocess_model(
                logl, lammodels, m0, vel, ccfconf=ccfconf)
            vsinisList.append(vel)
            res.append(cpa_model)
    return xlogl, res, retparams, vsinisList


def preprocess_data(lam, spec0, espec, ccfconf=None, badmask=None):
    """
    Preprocess data in the same manner as the template spectra, normalize by
    the continuum, apodize and pad

    Parameters:
    -----------
    lam: numpy array
        The wavelength vector
    spec0: numpy array
        The input spectrum vector
    espec0: numpy array
        The error-vector of the spectrum
    ccfconf: CCFConfig object
        The CCF configuration
    badmask: Numpy array(boolean), optional
        The optional mask for the CCF

    Returns:
    cap_spec: numpy array
        The processed apodized/normalized/padded spectrum

    """
    logl = np.linspace(ccfconf.logl0, ccfconf.logl1, ccfconf.npoints)
    curespec = espec.copy()
    curspec = spec0.copy()
    curespec[~badmask] = curespec[~badmask] * 0 + 1e9
    cont = get_continuum(lam, curspec, curespec, ccfconf=ccfconf)
    c_spec = spec0 / cont
    c_spec = c_spec - np.median(c_spec)
    ca_spec = apodize(c_spec)
    if badmask is not None:
        ca_spec[badmask] = 0
    ca_spec = scipy.interpolate.interp1d(np.log(lam), ca_spec, bounds_error=False,
                                         fill_value=0, kind='linear')(logl)
    lam1, cap_spec = pad(logl, ca_spec)
    return cap_spec


def ccf_executor(spec_setup, ccfconf, prefix=None, oprefix=None, every=10, vsinis=None):
    """
    Prepare the FFT transformations for the CCF

    Parameters:
    -----------
    spec_setup: string
        The name of the spectroscopic spec_setup
    ccfconf: CCFConfig
        The CCF configuration object
    prefix: string
        The input directory where the templates are located
    oprefix: string
        The output directory
    every: integer (optional)
        Produce FFTs of every N-th spectrum
    vsinis: list (optional)
        Produce FFTS of the templates  with Vsini from the list.
        Could be None (it means no rotation will be added)

    Returns:
    --------
    Nothing
    """

    postf = ''
    with open('%s/specs_%s%s.pkl' % (prefix, spec_setup, postf), 'rb') as fp:
        D = pickle.load(fp)
        vec, specs, lam, parnames = D['vec'], D['specs'], D['lam'], D['parnames']
        del D

    ndim = len(vec[:, 0])

    specs = specs[::every, :]
    vec = vec.T[::every, :]
    nspec, lenspec = specs.shape

    xlogl, models, params, vsinis = preprocess_model_list(
        lam, np.exp(specs), vec, ccfconf, vsinis=vsinis)
    ffts = np.array([np.fft.fft(x) for x in models])
    savefile = '%s/ccf_%s%s.pkl' % (oprefix, spec_setup, postf)
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
    ccfconf = make_ccf.CCFConfig(logl0=np.log(args.lambda0),
                                 logl1=np.log(args.lambda1),
                                 npoints=npoints)

    if args.vsinis is not None:
        vsinis = [float(_) for _ in args.vsinis.split(',')]
    else:
        vsinis = None
    ccf_executor(args.setup, ccfconf, args.prefix,
                 args.oprefix, args.every, vsinis)

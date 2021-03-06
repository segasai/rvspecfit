import pickle
import numpy as np
import scipy.optimize
import scipy.interpolate
from rvspecfit import make_ccf
import logging


class CCFCache:
    """ Singleton caching CCF information """
    ccf_info = {}
    ccfs = {}
    ccf_models = {}


def get_ccf_info(spec_setup, config):
    """
    Returns the CCF info from the pickled file for a given spectroscopic
    setup

    Parameters
    -----------
    spec_setup: string
        The spectroscopic setup needed
    config: dict
        The dictionary with the config

    Returns
    -------
    d: dict
        The dictionary with the CCF Information as saved by the make_ccf code

    """
    if spec_setup not in CCFCache.ccfs:
        prefix = config['template_lib']
        ccf_info_fname = prefix + make_ccf.CCF_PKL_NAME % spec_setup
        ccf_dat_fname = prefix + make_ccf.CCF_DAT_NAME % spec_setup
        ccf_mod_fname = prefix + make_ccf.CCF_MOD_NAME % spec_setup
        CCFCache.ccf_info[spec_setup] = pickle.load(open(ccf_info_fname, 'rb'))
        CCFCache.ccfs[spec_setup] = np.load(ccf_dat_fname, mmap_mode='r')
        CCFCache.ccf_models[spec_setup] = np.load(ccf_mod_fname, mmap_mode='r')
    return CCFCache.ccfs[spec_setup], CCFCache.ccf_models[
        spec_setup], CCFCache.ccf_info[spec_setup]


def ccf_combiner(ccfs):
    # combine ccfs from multiple filters
    # since ccf^2 is -chisq
    # we assume ccfs is 2d array shaped like Nfilters, nvelocities
    ret = (np.sign(ccfs) * ccfs**2).sum(axis=0)
    return ret


def fit(specdata, config):
    """
    Process the data by doing cross-correlation with templates

    Parameters
    -----------
    specdata: list of SpecData objects
        The list of data that needs to be fitted from differetn spectral
        setups.
    config: dict
        The configuration dictionary

    Returns
    results: dict
        The dictionary with results such as best template parameters,
        best velocity, best vsini.

    """
    # configuration parameters

    maxvel = config.get('max_vel') or 1000
    # only search for CCF peaks from -maxvel to maxvel
    nvelgrid = 2 * int(maxvel * 1. / (config.get('vel_step0') or 2)) + 1
    # number of points on the ccf in the specified velocity range
    vel_grid = np.linspace(-maxvel, maxvel, nvelgrid)

    # these are the dictionaries storing information for all the configurations
    # that we are fitting
    velstep = {}  # step in the ccf in velocity
    spec_fftconj = {}  # conjugated fft of the data
    vels = {}  # velocity grids
    subind = {}  # the range of the ccf covering the velocity range of interest
    ccf_dats = {}  # ffts of templates
    ccf_infos = {}  # ccf configurations
    ccf_mods = {}  # the actual template models
    proc_specs = {}  # actual data processed/continuum normalized etc
    setups = []
    for cursd in specdata:
        spec_setup = cursd.name
        setups.append(spec_setup)
        lam = cursd.lam
        spec = cursd.spec
        espec = cursd.espec
        ccf_dats[spec_setup], ccf_mods[spec_setup], ccf_infos[
            spec_setup] = get_ccf_info(spec_setup, config)
        ccfconf = ccf_infos[spec_setup]['ccfconf']
        logl0 = ccfconf.logl0
        logl1 = ccfconf.logl1
        npoints = ccfconf.npoints
        proc_spec = make_ccf.preprocess_data(lam,
                                             spec,
                                             espec,
                                             badmask=cursd.badmask,
                                             ccfconf=ccfconf)
        proc_spec_std = proc_spec.std()
        if proc_spec_std == 0:
            proc_spec_std = 1
            logging.warning('Spectrum looks like a constant...')
        proc_spec /= proc_spec_std
        proc_specs[spec_setup] = proc_spec
        spec_fft = np.fft.fft(proc_spec)
        spec_fftconj[spec_setup] = spec_fft.conj()
        cur_step = (np.exp((logl1 - logl0) / npoints) - 1) * 3e5
        lspec = len(spec_fft)
        cur_off = lspec // 2
        # this is the wrapping point
        cur_vels = -((np.arange(lspec) + cur_off) % lspec - cur_off) * cur_step
        # now cur_vels[lspec-off] is the first positive velocity
        # we need to np.roll(X,cur_off)  to make it continuous
        # notice that it is decreasing and it corresponds to the velocity of
        # the ccf pixels
        cur_ind = (np.abs(cur_vels) < (maxvel + cur_step))
        # boolean mask within the required velocity range
        assert (cur_ind.sum() % 2 == 1)  # must be odd
        cur_ind = np.roll(np.nonzero(cur_ind)[0], cur_ind.sum() // 2)
        # these are indices that makes it monotonic
        cur_ind = cur_ind[::-1]
        # that provides indices that will go from negative
        # to positive velocities
        subind[spec_setup] = cur_ind
        velstep[spec_setup] = cur_step
        vels[spec_setup] = cur_vels[cur_ind]

    max_ccf = -np.inf
    best_id = -1

    # the logic is the following
    # if array y is shifted by n pixels to the right side wrt x
    # ifft(fft(x)*fft(y).conj) will peak at pixel N-n (0based)
    # or if array is shifted to n pixels to the left it will peak at n (0based)

    nfft = ccf_dats[spec_setup].shape[0]
    curccf = np.empty((len(setups), nvelgrid))
    for cur_id in range(nfft):

        for ii, spec_setup in enumerate(setups):
            curf = ccf_dats[spec_setup][cur_id, :]
            curccf0 = np.fft.ifft(spec_fftconj[spec_setup] * curf).real
            curccf[ii] = scipy.interpolate.interp1d(
                vels[spec_setup],
                curccf0[subind[spec_setup]],
            )(vel_grid)
            # we interpolate all the ccf from every arm
            # to the same velocity grid

        allccf = ccf_combiner(curccf)
        if allccf.max() > max_ccf:
            max_ccf = allccf.max()
            best_id = cur_id
            best_vel = vel_grid[np.argmax(allccf)]
            best_ccf = allccf

    if best_id < 0:
        logging.error('Cross-correlation failed')
        raise RuntimeError('Cross-correlation step failed')

    best_model = {}
    for spec_setup in setups:
        best_model[spec_setup] = np.roll(ccf_mods[spec_setup][best_id],
                                         int(best_vel / velstep[spec_setup]))
    best_par = ccf_infos[setups[0]]['params'][best_id]
    best_par = dict(zip(ccf_infos[setups[0]]['parnames'], best_par))
    best_vsini = ccf_infos[setups[0]]['vsinis'][best_id]

    result = dict(best_par=best_par,
                  best_vel=best_vel,
                  best_ccf=best_ccf,
                  best_vsini=best_vsini,
                  best_model=best_model,
                  proc_spec=proc_specs)

    return result

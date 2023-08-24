import pickle
import numpy as np
import scipy.optimize
import scipy.interpolate
from rvspecfit import make_ccf
from rvspecfit.spec_fit import SpecData
import logging


class CCFCache:
    """ Singleton caching CCF information """
    ccf_info = {}
    ccfs = {}
    ccf2s = {}
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
        ccf_continuum = config.get('ccf_continuum_normalize')
        if ccf_continuum is None:
            ccf_continuum = True
        ccf_info_fname = prefix + make_ccf.get_ccf_pkl_name(
            spec_setup, ccf_continuum)
        ccf_dat_fname = prefix + make_ccf.get_ccf_dat_name(
            spec_setup, ccf_continuum)
        ccf_mod_fname = prefix + make_ccf.get_ccf_mod_name(
            spec_setup, ccf_continuum)
        CCFCache.ccf_info[spec_setup] = pickle.load(open(ccf_info_fname, 'rb'))
        C = np.load(ccf_dat_fname, mmap_mode='r')
        CCFCache.ccfs[spec_setup] = C['fft']
        CCFCache.ccf2s[spec_setup] = C['fft2']
        CCFCache.ccf_models[spec_setup] = np.load(ccf_mod_fname, mmap_mode='r')
    return CCFCache.ccfs[spec_setup], CCFCache.ccf2s[
        spec_setup], CCFCache.ccf_models[spec_setup], CCFCache.ccf_info[
            spec_setup]


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
    ivar_fftconj = {}  # conjugated fft of the data
    vels = {}  # velocity grids
    subind = {}  # the range of the ccf covering the velocity range of interest
    ccf_dats = {}  # ffts of templates
    ccf2_dats = {}  # ffts of templates
    ccf_infos = {}  # ccf configurations
    ccf_mods = {}  # the actual template models
    proc_specs = {}  # actual data processed/continuum normalized etc
    proc_ivars = {}
    setups = []
    ccf_confs = []
    if isinstance(specdata, SpecData):
        # if we got a single one put it in the list
        specdata = [specdata]
    for cursd in specdata:
        spec_setup = cursd.name
        setups.append(spec_setup)
        lam = cursd.lam
        spec = cursd.spec
        espec = cursd.espec
        (ccf_dats[spec_setup], ccf2_dats[spec_setup], ccf_mods[spec_setup],
         ccf_infos[spec_setup]) = get_ccf_info(spec_setup, config)
        ccfconf = ccf_infos[spec_setup]['ccfconf']
        ccf_confs.append(ccfconf)
        logl0 = ccfconf.logl0
        logl1 = ccfconf.logl1
        npoints = ccfconf.npoints
        proc_spec, proc_ivar = make_ccf.preprocess_data(lam,
                                                        spec,
                                                        espec,
                                                        badmask=cursd.badmask,
                                                        ccfconf=ccfconf)
        proc_specs[spec_setup] = proc_spec
        proc_ivars[spec_setup] = proc_ivar
        spec_fft = np.fft.rfft(proc_spec * proc_ivar)
        ivar_fft = np.fft.rfft(proc_ivar)

        spec_fftconj[spec_setup] = spec_fft.conj()
        ivar_fftconj[spec_setup] = ivar_fft.conj()

        cur_step = (np.exp((logl1 - logl0) / npoints) - 1) * 3e5
        lspec = len(proc_spec)
        # Importantly this has to be length of the spectrum rather than
        # length of the fft since we are using rfft
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

    # the logic is the following
    # if array y is shifted by n pixels to the right side wrt x
    # ifft(fft(x)*fft(y).conj) will peak at pixel N-n (0based)
    # or if array is shifted to n pixels to the left it will peak at n (0based)

    nfft = ccf_dats[spec_setup].shape[0]

    all_chisqs = []
    for cur_id in range(nfft):
        cur_chisq = np.zeros(len(vel_grid))
        for ii, spec_setup in enumerate(setups):
            curf = ccf_dats[spec_setup][cur_id, :]
            curf2 = ccf2_dats[spec_setup][cur_id, :]
            curccf0 = np.fft.irfft(spec_fftconj[spec_setup] * curf)
            curccf1 = np.fft.irfft(ivar_fftconj[spec_setup] * curf2)
            # chisquare i -2* l * S/E^2 xx T + l^2 1/E^2 xx T^2
            #  l is the multiplier (S-lT)
            # the best value is l = ((S/E^2) xx T)/(1/E^2 xx T^2)
            # Thus  the best chisq is -((S/E^2) xx T)^2/(1/E^2 xx T^2)
            # where xx is the convolution operator
            if ccf_confs[ii].continuum:
                chisq = -2 * curccf0 + curccf1
            else:
                chisq = (-curccf0**2 / curccf1)
            cur_chisq += (scipy.interpolate.interp1d(vels[spec_setup],
                                                     chisq[subind[spec_setup]],
                                                     kind='linear')(vel_grid))
            # we interpolate all the ccf from every arm
            # to the same velocity grid
        all_chisqs.append(cur_chisq)

    all_chisqs = np.array(all_chisqs)
    best_id = np.argmin(all_chisqs.min(axis=1))
    best_ccf = all_chisqs[best_id]
    best_pix = np.argmin(best_ccf)
    best_vel = vel_grid[best_pix]

    if not np.isfinite(all_chisqs[best_id, best_pix]):
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
                  proc_spec=proc_specs,
                  vel_grid=vel_grid)
    return result

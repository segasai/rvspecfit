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
    Returns the CCF info from the pickled file for a given spectroscopic spec_setup

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
    ret = 0
    for curc in ccfs:
        ret = ret + np.sign(curc) * curc**2
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
        The dictionary with results such as best template parameters, best velocity
        best vsini.

    """
    # configuration parameters

    maxvel = config.get('max_vel') or 1000
    # only search for CCF peaks from -maxvel to maxvel
    nvelgrid = int(2 * maxvel / (config.get('vel_step0') or 2))
    # number of points on the ccf in the specified velocity range

    loglam = {}
    velstep = {}
    spec_fftconj = {}
    vels = {}
    off = {}
    subind = {}
    ccf_dats = {}
    ccf_infos = {}
    ccf_mods = {}
    proc_specs = {}
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
        velstep[spec_setup] = (np.exp((logl1 - logl0) / npoints) - 1) * 3e5
        l = len(spec_fft)
        off[spec_setup] = l // 2
        vels[spec_setup] = ((np.arange(l) + off[spec_setup]) % l -
                            off[spec_setup]) * velstep[spec_setup]
        vels[spec_setup] = -np.roll(vels[spec_setup], off[spec_setup])
        subind[spec_setup] = np.abs(vels[spec_setup]) < maxvel

    maxv = -1e20
    best_id = -90
    best_v = -777

    nfft = ccf_dats[spec_setup].shape[0]
    vel_grid = np.linspace(-maxvel, maxvel, nvelgrid)
    best_ccf = vel_grid * 0
    for id in range(nfft):
        curccf = {}
        for spec_setup in setups:
            curf = ccf_dats[spec_setup][id, :]
            ccf = np.fft.ifft(spec_fftconj[spec_setup] * curf).real
            ccf = np.roll(ccf, off[spec_setup])
            curccf[spec_setup] = ccf[subind[spec_setup]]
            curccf[spec_setup] = scipy.interpolate.UnivariateSpline(
                vels[spec_setup][subind[spec_setup]][::-1],
                curccf[spec_setup][::-1],
                s=0)(vel_grid)

        allccf = ccf_combiner([curccf[_] for _ in setups])
        if allccf.max() > maxv:
            maxv = allccf.max()
            best_id = id
            best_v = vel_grid[np.argmax(allccf)]
            best_model = {}
            for spec_setup in setups:
                best_model[spec_setup] = np.roll(
                    ccf_mods[spec_setup][id],
                    int(best_v / velstep[spec_setup]))
            best_ccf = allccf
    try:
        assert (best_id >= 0)
    except:
        raise Exception('Cross-correlation step failed')
    best_par = ccf_infos[setups[0]]['params'][best_id]
    best_par = dict(zip(ccf_infos[setups[0]]['parnames'], best_par))

    best_vsini = ccf_infos[setups[0]]['vsinis'][best_id]

    result = {}
    result['best_par'] = best_par
    result['best_vel'] = best_v
    result['best_ccf'] = best_ccf
    result['best_vsini'] = best_vsini
    result['best_model'] = best_model
    result['proc_spec'] = proc_specs
    return result

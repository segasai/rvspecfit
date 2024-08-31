from os import environ, getpid, rename, makedirs

environ["OMP_NUM_THREADS"] = "1"

from socket import gethostname
from os.path import isdir, exists
from time import sleep, time, ctime
from logging import warning, info, error, debug, exception, basicConfig
from operator import lt, gt
from enum import Enum
from importlib import import_module
from sys import version, argv, exit
from dateutil.parser import isoparse
from traceback import print_exc
from concurrent.futures import ProcessPoolExecutor
from argparse import ArgumentParser

from matplotlib.pyplot import (clf, figure, plot, fill_between, gca,
                               title, xlabel, savefig, close)
from matplotlib import rcParams
from scipy.stats import scoreatpercentile
from astropy.wcs import WCS
from astropy import units as u
from astropy.io.fits import (Header, PrimaryHDU, ImageHDU, BinTableHDU,
                             HDUList)
from astropy.io.fits import open as fits_open
from astropy.time import Time
from astropy.table import Table
import numpy as np
from rvspecfit.spec_fit import SpecData, get_chisq_continuum 
from rvspecfit.fitter_ccf import fit
from rvspecfit.vel_fit import firstguess, process
from rvspecfit.spec_inter import interp_cache
from rvspecfit.utils import read_config
from rvspecfit import _version

class FileQueue:
    """
    This is a class that can work as an iterator.
    Here we can either provide the list of files or the file with the
    list of files or use it as queue, where we pick up the top file,
    remove the line from the file and move on
    """
    def __init__(self, file_list=None, file_from=None, queue=False):
        if file_list:
            self.file_list = file_list
            self.file_from = None
            self.queue = False
        elif file_from:
            if not queue:
                self.file_list = []
                self.file_from = file_from
                with open(file_from, "r") as file_obj:
                    for ll in file_obj:
                        self.file_list.append(ll.rstrip())
            else:
                self.file_list = None
                self.file_from = file_from
                self.queue = queue

    def __iter__(self):
        return self

    def __next__(self):
        if self.file_list:
            if len(self.file_list) > 0:
                return self.file_list.pop(0)
            else:
                raise StopIteration
        else:
            return self.read_next()

    def read_next(self):
        lockname = (
            "{}.{}.{:d}.lock".format(self.file_from, gethostname(),
                                     getpid())
        )
        wait_time = 1
        max_waits = 1000
        for i in range(max_waits):
            try:
                rename(self.file_from, lockname)
            except FileNotFoundError:
                sleep(np.random.uniform(wait_time, 1.5 * wait_time))
                continue
            try:
                with open(lockname, "r") as file_obj:
                    ll = file_obj.readlines()
                if len(ll) == 0:
                    raise StopIteration
                ret = ll[0].rstrip()
                with open(lockname, "w") as file_obj:
                    file_obj.writelines(ll[1:])
                return ret
            finally:
                rename(lockname, self.file_from)

        warning("Cannot read next file due to lock")
        raise StopIteration


class FakeFuture(object):
    # this is a fake Future object designed for easier switching
    # to single thread operations when debugging
    def __init__(self, func):
        self.func = func

    def result(self):
        return self.func


class FakeExecutor(object):
    def __init__(self):
        pass

    def submit(self, func, *tuple_args, **dct_kwargs):
        return FakeFuture(func(*tuple_args, **dct_kwargs))


def upd_proc_status_file(status_filename, proc_file, status, no_objs,
                         time_sec, start=False):
    if start:
        with open(status_filename, "w") as file_obj:
            pass
        if proc_file:
            pass
        else:
            return
    with open(status_filename, "a") as file_obj:
        print(
            "{} {} {} {:.2f}".format(
                proc_file, status, no_objs, time_sec),
            file=file_obj)
    return


def make_plot(
    lst_spec_data_objs, lst_arr_fluxess_phoenix_syn_stellar_spec_mods,
    str_title, fig_filename):
    """
    Make a plot with the spectra and fits

    Parameters
    ----------
    lst_spec_data_objs : List of SpecData objects
        The object with the spectral data
    lst_arr_fluxess_phoenix_syn_stellar_spec_mods : list of numpy arrays
        The list of arrays of fluxes of fit models
    str_title : string
        The figure title
    fig_filename : string
        The filename of the figure
    """  
    clf()
    no_panels = len(lst_spec_data_objs)
    figsize_tuple = (10., 3. * no_panels)
    fig_obj = figure(figsize=(figsize_tuple), dpi=300.)
    for i in range(no_panels):
        fig_obj.add_subplot(no_panels, 1, i + 1)
        spec_data_obj = lst_spec_data_objs[i].spec
        perc = .2
        xind_arr = lst_spec_data_objs[i].badmask
        ymin, ymax = [
            scoreatpercentile(
                spec_data_obj[np.logical_not(xind_arr)], _)
            for _ in [perc, 100. - perc]
        ]
        plot(lst_spec_data_objs[i].lam, lst_spec_data_objs[i].spec,
             c="k", linewidth=rcParams["xtick.minor.width"])
        plot(
            lst_spec_data_objs[i].lam[xind_arr],
            lst_spec_data_objs[i].spec[xind_arr], "b.",
            linewidth=rcParams["xtick.minor.width"])
        fill_between(lst_spec_data_objs[i].lam,
                     lst_spec_data_objs[i].spec
                     - lst_spec_data_objs[i].espec,
                     lst_spec_data_objs[i].spec
                     + lst_spec_data_objs[i].espec,
                     color="gray", alpha=.1, zorder=2)
        plot(
            lst_spec_data_objs[i].lam,
            lst_arr_fluxess_phoenix_syn_stellar_spec_mods[i], c="r",
            alpha=.8, linewidth=rcParams["xtick.minor.width"])
        gca().set_ylim(ymin, ymax)
        if i:
            pass
        elif i == no_panels - 1:
            title(str_title)
            xlabel(r"$\lambda$ [$\AA$]")
        else:
            continue
    savefig(fig_filename)
    close(fig_obj)
    
    
def read_hdul_w_data(hdul):
    """ Read the data file

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        List of Header Data Units in which data of spectrum is stored 

    Returns
    -------
    arr_fluxes: ndarray
        1d array of fluxes
    arr_wavs: ndarray
        1d array of wavelengths
    arr_dfluxes : ndarray
        1d array of uncertainties of fluxes
    """
    arr_fluxes = hdul[1].data["flux"]
    arr_wavs = 10.**hdul[1].data["loglam"]
    arr_dfluxes = np.sqrt(1./hdul[1].data["ivar"])
    return arr_fluxes, arr_wavs, arr_dfluxes
        
    
def get_lst_spec_data_objs(arr_wavs, arr_fluxes, arr_dfluxes,
                           config_name):
    """ Return the list of SpecData instances for one single object

    Parameters
    ----------
    arr_wavs : ndarray
        1d wavelength array
    arr_fluxes : ndarray
        1d flux array
    arr_dfluxes : ndarray
        1d array of flux uncertainties
    config_name : str
        Configuration name

    Returns
    -------
    lst_spec_data_objs : list
        List of SpecData objects or None if failed

    """
    large_error = 1000
    lst_spec_data_objs = []
    # If the uncertainty is smaller than this times median uncertainty,
    # clamp the uncertainty
    min_unc_frac = .3
    ivar_arr_fluxes = 1. / arr_dfluxes**2. 
    median_flux = np.nanmedian(arr_fluxes)
    if median_flux:
        pass
    else:
        median_flux = np.nanmedian(arr_fluxes[arr_fluxes > 0])
        if np.isfinite(median_flux):
            pass
        else:
            median_flux = np.nanmedian(np.abs(arr_fluxes))
    if np.isfinite(median_flux) or median_flux:
        inf_arr_fluxes_msk = ~np.isfinite(arr_fluxes + ivar_arr_fluxes)
        arr_wavs_msk = (arr_wavs<3900.) & (arr_wavs>5990.)
        ivar_arr_fluxes_msk = ivar_arr_fluxes <= 0.
        msk_arr = inf_arr_fluxes_msk | ivar_arr_fluxes_msk | arr_wavs_msk
        ivar_arr_fluxes[msk_arr] = 1. / median_flux**2. / large_error**2.
        arr_fluxes[msk_arr] = median_flux
        arr_dfluxes = 1. / ivar_arr_fluxes**.5
        if msk_arr.all():
            warning("The whole spectrum was masked...")
        else:
            arr_dfluxes_good = arr_dfluxes[~msk_arr]
            dflux_thresh = np.median(arr_dfluxes_good) * min_unc_frac
            replace_idx_arr = (arr_dfluxes<dflux_thresh) & (~msk_arr)
            if replace_idx_arr.sum() / (~msk_arr).sum() > .1:
                warning(
                    "More than 10% of spectra had the uncertainty"
                    " clamped down")
            arr_dfluxes[replace_idx_arr] = dflux_thresh

        spec_data_obj = SpecData(config_name, arr_wavs, arr_fluxes,
                                 arr_dfluxes, msk_arr)
        lst_spec_data_objs.append(spec_data_obj)
        return lst_spec_data_objs
    else:
        # Bail out the spectrum, if it is insane
        return None
        
        
def proc_spec_data_obj(lst_spec_data_objs, config_dct, dct_options,
                       fig_filename, ccfinit=True, doplot=True,
                       dct_priors=None):
    """ Process one single SpecData object

    Parameters
    ----------
    lst_spec_data_objs : list
        List of SpecData objects to be fit
    config_dct : dict
        Dictionary with configuration
    dct_options : dict
        Configuration options
    fig_filename : str
        Filename for the plot
    ccfinit : bool, default True
        If true, the starting point for the fit will be determined
        through cross-correlation rather than brute-force grid search
    doplot : bool, default True
        Do plotting or not
    dct_priors : dict (optional)
        Extra dictionary with normal priors on paramaters, i.e., 
        {"teff":(5000, 10)} for N(5000, 10) prior on the
        effective temperature in Kelvin
                       
    Returns
    -------
    outdct : dict
        Dictionary with fit parameters
    lst_arr_fluxess_phoenix_syn_stellar_spec_mods : list
        List of arrays with fluxes of best-fit PHOENIX synthetic stellar
        spectrum models
    """
    dct_chisqs = {}
    dct_chisqs_c = {}
    time1 = time()
    if ccfinit:
        dct_res = fit(lst_spec_data_objs, config_dct)
        dct_res["template_pars_mx_cross_corr"] = (
            dct_res["best_par"])
        del dct_res["best_par"]
        dct_template_ini_par_values = (
            dct_res["template_pars_mx_cross_corr"])
    else:
        dct_res = firstguess(lst_spec_data_objs, config=dct_config,
                             options=dct_options)
        dct_res["vsini_kms"] = dct_res["vsini"]
        del dct_res["vsini"]
        dct_res["best_vsini"] = dct_res.get("vsini_kms")
        dct_template_ini_par_values = dct_res
        dct_template_ini_par_values["vsini"] = (
            dct_template_ini_par_values["best_vsini"])
        del dct_template_ini_par_values["best_vsini"]
    time2 = time()
    lst_fix_params = []

    if dct_res["best_vsini"] is not None:
        dct_template_ini_par_values["vsini"] = np.clip(
            dct_res["best_vsini"], config_dct["min_vsini"],
            config_dct["max_vsini"])

    dct_res_mle = process(
        lst_spec_data_objs, dct_template_ini_par_values,
        fixParam=lst_fix_params, config=config_dct,
        options=dct_options, priors=dct_priors)
    dct_res_mle["vel_kms"] = dct_res_mle["vel"]
    del dct_res_mle["vel"]
    dct_res_mle["dvel_kms"] = dct_res_mle["vel_err"]
    del dct_res_mle["vel_err"]
    dct_res_mle["phoenix_syn_spec_params"] = dct_res_mle["param"]
    del dct_res_mle["param"]
    dct_res_mle["phoenix_syn_spec_params"]["lg_g"] = (
        dct_res_mle["phoenix_syn_spec_params"]["logg"])
    del dct_res_mle["phoenix_syn_spec_params"]["logg"]
    dct_res_mle["phoenix_syn_spec_params"]["eff_temp_K"] = (
        dct_res_mle["phoenix_syn_spec_params"]["teff"])
    del dct_res_mle["phoenix_syn_spec_params"]["teff"]
    dct_res_mle["phoenix_syn_spec_params"]["alphafe"] = (
        dct_res_mle["phoenix_syn_spec_params"]["alpha"])
    del dct_res_mle["phoenix_syn_spec_params"]["alpha"]
    dct_res_mle["unc_phoenix_syn_spec_params"] = (
        dct_res_mle["param_err"])
    del dct_res_mle["param_err"]
    dct_res_mle["unc_phoenix_syn_spec_params"]["lg_g"] = (
        dct_res_mle["unc_phoenix_syn_spec_params"]["logg"])
    del dct_res_mle["unc_phoenix_syn_spec_params"]["logg"]
    dct_res_mle["unc_phoenix_syn_spec_params"]["eff_temp_K"] = (
        dct_res_mle["unc_phoenix_syn_spec_params"]["teff"])
    del dct_res_mle["unc_phoenix_syn_spec_params"]["teff"]
    dct_res_mle["unc_phoenix_syn_spec_params"]["alphafe"] = (
        dct_res_mle["unc_phoenix_syn_spec_params"]["alpha"])
    del dct_res_mle["unc_phoenix_syn_spec_params"]["alpha"]
    dct_res_mle["lst_arr_fluxes_phoenix_syn_stellar_spec_mod"] = (
        dct_res_mle["yfit"])
    del dct_res_mle["yfit"]
    time3 = time()
    chisq_cont_array = (
        get_chisq_continuum(
            lst_spec_data_objs, options=dct_options
        )["chisq_array"]
    )
    time4 = time()
    outdct = {
        "vel":dct_res_mle["vel_kms"] * u.km / u.s,
        "dvel":dct_res_mle["dvel_kms"] * u.km / u.s,
        "vel_skew":dct_res_mle["vel_skewness"],
        "vel_kurt":dct_res_mle["vel_kurtosis"],
        "lg_g":dct_res_mle["phoenix_syn_spec_params"]["lg_g"],
        "eff_temp":(
            dct_res_mle["phoenix_syn_spec_params"]["eff_temp_K"] * u.K),
        "alphafe":dct_res_mle["phoenix_syn_spec_params"]["alphafe"],
        "feh":dct_res_mle["phoenix_syn_spec_params"]["feh"],
        "dlg_g":dct_res_mle["unc_phoenix_syn_spec_params"]["lg_g"],
        "deff_temp":(
            dct_res_mle["unc_phoenix_syn_spec_params"]["eff_temp_K"]
            * u.K),
        "dalpha_fe":(
            dct_res_mle["unc_phoenix_syn_spec_params"]["alphafe"]),
        "dfeh":dct_res_mle["unc_phoenix_syn_spec_params"]["feh"],
        "vsini":dct_res_mle["vsini"] * u.km / u.s}

    for i, spec_data_obj in enumerate(lst_spec_data_objs):
        if spec_data_obj.name not in dct_chisqs:
            dct_chisqs[spec_data_obj.name] = 0
            dct_chisqs_c[spec_data_obj.name] = 0
        dct_chisqs[spec_data_obj.name] += dct_res_mle["chisq_array"][i]
        dct_chisqs_c[spec_data_obj.name] += chisq_cont_array[i]

    outdct["tot_chisq"] = sum(dct_chisqs.values())
    outdct["tot_chisq_c"] = sum(dct_chisqs_c.values())

    for config_name in dct_chisqs.keys():
        outdct["chisq_{}".format(config_name)] = dct_chisqs[config_name]
        outdct["chisq_c_{}".format(config_name)] = float(
            dct_chisqs_c[config_name])

    chisq_thresh = 50
    # If the delta-chisq between continuum is smaller than this, we set
    # a warning flag if [Fe/H] is close to the edge of the possible
    # values.
    feh_thresh = .01
    lst_feh_edges = [-4., 1.]

    eff_temp_thresh = 10.
    lst_eff_temp_edges = [2300., 15000.]

    # If we are within this threshold of the velocity boundary, we set
    # another warning.
    vel_edge_thresh = 5.

    dvel_thresh = 100.
    # If the uncertainty is larger than this, we warn.

    vel_warn = 0
    dct_bitmasks = {"chisq_warn":1, "vel_warn":2, "dvel_warn":4,
                    "param_warn":8}
    dchisq = outdct["tot_chisq_c"] - outdct["tot_chisq"]  # Should be >0

    if (dchisq < chisq_thresh):
        vel_warn |= dct_bitmasks["chisq_warn"]
    cur_vel = outdct["vel"].to_value(u.km / u.s)
    if (cur_vel > config_dct["max_vel"] - vel_edge_thresh
        or cur_vel < config_dct["min_vel"] + vel_edge_thresh):
        vel_warn |= dct_bitmasks["vel_warn"]

    if (outdct["dvel"].to_value(u.km / u.s) > dvel_thresh):
        vel_warn |= dct_bitmasks["dvel_warn"]

    for cur_param_name, lst_cur_edges, cur_thresh in [
        ["eff_temp_K", lst_eff_temp_edges, eff_temp_thresh],
        ["feh", lst_feh_edges, feh_thresh]]:
        for xid, cur_oper in [[0, lt], [1, gt]]:
            # If for left edge we are doing <, i.e, the value is less
            # than the left edge, this is bad
            # 
            # for right we are doing >
            if xid == 0:
                # left_edge shift to the right
                cur_val = lst_cur_edges[xid] + cur_thresh
            if xid == 1:
                # right edge shift to the left
                cur_val = lst_cur_edges[xid] - cur_thresh
            if cur_oper(
                dct_res_mle["phoenix_syn_spec_params"][cur_param_name],
                cur_val):
                vel_warn |= dct_bitmasks["param_warn"]
    outdct["vel_warn"] = vel_warn

    if doplot:
        str_title = (
            "".join(
                ["lg(g/cm/s2) = {:.1f} Teff = {:.1f} K [Fe/H]".format(
                     dct_res_mle["phoenix_syn_spec_params"]["lg_g"],
                     dct_res_mle["phoenix_syn_spec_params"][
                         "eff_temp_K"]
                 ),
                 " = {:.1f} [alpha/Fe] = {:.1f} velocity = ".format(
                     dct_res_mle["phoenix_syn_spec_params"]["feh"],
                     dct_res_mle["phoenix_syn_spec_params"]["alphafe"]),
                 " = {:.1f} +/- {:.1f} km/s v sin i = {:.1f}".format(
                     dct_res_mle["vel_kms"], dct_res_mle["dvel_kms"],
                     dct_res_mle['vsini']),
                 " km/s"
                ]
            )
        )
        make_plot(
            lst_spec_data_objs,
            dct_res_mle["lst_arr_fluxes_phoenix_syn_stellar_spec_mod"],
            str_title, fig_filename)
    dct_versions = {}
    for i, (k, v) in enumerate(interp_cache.interps.items()):
        dct_versions[k] = {
            "revision":v.revision,
            "creation_soft_version":v.creation_soft_version}
    outdct["vers"] = dct_versions
    debug(
        "Timing: {:.4f} {:.4f} {:.4f}".format(
            time2 - time1, time3 - time2, time4 - time3)
    )
    return (outdct,
            dct_res_mle["lst_arr_fluxes_phoenix_syn_stellar_spec_mod"])


class ProcessStatus(Enum):
    SUCCESS = 0
    FAILURE = 1
    EXISTING = 2

    def __str__(self):
        return self.name


def get_dep_versions():
    """
    Get versions of required packages
    """
    lst_pkg_names = ["numpy", "astropy", "matplotlib", "rvspecfit",
                     "pandas", "scipy", "yaml", "numdifftools"]
    # Ideally you need to check that the list here matches the 
    # requirements.txt
    ret_dct = {}
    for pkg_name in lst_pkg_names:
        ret_dct[pkg_name] = import_module(pkg_name).__version__
    ret_dct["python"] = str.split(version, " ")[0]
    return ret_dct


def get_prim_header(dct_versions={}, config_dct=None, cmdline=None,
                    spectrum_header=None, dct_priors=None):
    """
    Return the primary Header Data Unit with various info in the header
    """
    header = Header()
    for i, (pkg_name, pkg_v) in enumerate(get_dep_versions().items()):
        header["depnam{:02d}".format(i)] = (pkg_name, "Software")
        header["depver{:02d}".format(i)] = (pkg_v, "Version")
    for i, (k, v) in enumerate(dct_versions.items()):
        header["tmplrev{:d}".format(i)] = (
            v["revision"], "Spec template revision")
        header["tmplsvr{:d}".format(i)] = (
            v["creation_soft_version"], "Spec template soft version")
    if config_dct:
        header["vel_conf"] = config_dct["config_file_path"]
    if cmdline:
        header["vel_cmd"] = cmdline

    # keywords to copy from the header of the spectrum
    lst_cp_keys = ["MJD"]

    if spectrum_header:
        for key in lst_cp_keys:
            if key in spectrum_header:
                header[key] = spectrum_header[key]
                
    for par_name in dct_priors:
        header["{}_normal_prior".format(par_name)] = (
            "N({}, {})".format(
                dct_priors[par_name][0], dct_priors[par_name][1]),
            "Normal prior of {}".format(par_name))
    return header


def get_column_desc():
    """
    Return the list of column descriptions when we fitting a given set
    of configurations
    """
    dct_column_desc = {
        "vel":"Velocity in km/s", "dvel":"Velocity uncertainty in km/s",
        "vel_skew":"Velocity posterior skewness",
        "vel_kurt":"Velocity posterior kurtosis",
        "vsini":"Stellar rotation velocity in km/s",
        "lg_surf_grav":"Logarithm of surface gravity",
        "teff":"Effective temperature in K",
        "feh":"[Fe/H] from template fitting",
        "alphafe":"[alpha/Fe] from template fitting",
        "dlg_surf_grav":"Uncertainty of logarithm of surface gravity",
        "dteff":"Effective temperature uncertainty in K",
        "dfeh":"[Fe/H] uncertainty from template fitting",
        "dalphafe":"[alpha/Fe] uncertainty from template fitting",
        "tot_chisq":"Total chi-square",
        "tot_chisq_c":"Total chi-square for polynomial only fit",
        "name":"Name of target",
        "mjd":"Modified Julian days of observation",
        "success":"Did we succeed or fail",
        "vel_warn":"RVSpecFit warning flag"}
    return dct_column_desc


def comment_filler(bin_tab_hdu, desc):
    """
    Fill comments in the Flexible Image Transport System file header
    """
    for i, bin_tab_column_name in enumerate(bin_tab_hdu.data.columns):
        bin_tab_hdu.header["tcomm{}".format(i + 1)] = (
            desc.get(bin_tab_column_name.name) or "")
    return bin_tab_hdu


def write_hdulist(filename, hdulist):
    """
    Write HDUList to file using a temporary file which is then renamed
    """
    filename_tmp = filename + ".tmp"
    hdulist.writeto(filename_tmp, overwrite=True, checksum=True)
    rename(filename_tmp, filename)
                                            
    
def proc_file_w_spec(name_file_w_spec_data, config_name, fig_prefix,
                     config_dct, mod_ofilename, tab_ofilename, npoly=10,
                     wav0=None, wav1=None, doplot=True, ex_obj=None,
                     ccfinit=True, cmdline=None, dct_priors=None):
    """
    Process one single file with Sloan Digital Sky Survey spectra  

    Parameters
    -----------
    name_file_w_spec_data : str
        The filename with the spectra to be fitted
    config_name : str
        Configuration name
    fig_prefix: str
        The prefix where the figures will be stored
    config_dct : Dictionary
        The configuration dictionary
    mod_ofilename : str
        The filename where the model will be stored
    tab_ofilename : str
        The filename where the table with parameters will be stored
    npoly : integer, default 10
        The degree of the polynomial used for continuum
    wav0 : float, default None
        Wavelength of the lower limit of the considered range across
        dispersion axis of spectrum (should be in same units as unit
        used for wavelengths of dispersion axis)
    wav1 : float, default None
        Wavelength of the upper limit of the considered range across
        dispersion axis of spectrum (should be in same units as unit
        used for wavelengths of dispersion axis)
    doplot : bool, default True
        Produce plots
    ex_obj : Executor
         The executor that will run parallel processes
    ccfinit : bool, default True
         If true (default the starting point will be determined from
         cross-correlation as opposed to brute-force grid search
    cmdline : string, default None
        The command line used in execution of RvSpecFit
    dct_priors : dict (optional)
        Extra dictionary with normal priors on paramaters, i.e., 
        {"teff":(5000, 10)} for N(5000, 10) prior on the
        effective temperature in Kelvin
    """
    if npoly:
        pass
    else:
        npoly = 10
    dct_options = {"npoly":npoly}
    lst_times = []
    lst_times.append(time())
    info("Processing {}".format(name_file_w_spec_data))
    try:
        hdul_w_spec_data = fits_open(name_file_w_spec_data)
    except OSError:
        error("Cannot read file {}".format(name_file_w_spec_data))
        return -1

    fits_file_hdr = hdul_w_spec_data[0].header

    arr_fluxes, arr_wavs, arr_dfluxes = read_hdul_w_data(hdul_w_spec_data)
    hdul_w_spec_data.close()
    
    # columns to include in the RVTAB
    lst_columns_copy = ["name", "mjd"]

    lst_outdf = []

    lst_times.append(time())
    lst_tuple_rets = []
    # Collect data
    if wav0 and wav1:
        lst_spec_data_objs = get_lst_spec_data_objs(
            arr_wavs[(arr_wavs>wav0) & (arr_wavs<wav1)],
            arr_fluxes[(arr_wavs>wav0) & (arr_wavs<wav1)],
            arr_dfluxes[(arr_wavs>wav0) & (arr_wavs<wav1)], config_name)
    else:
        lst_spec_data_objs = get_lst_spec_data_objs(arr_wavs, arr_fluxes,
                                                    arr_dfluxes,
                                                    config_name)
    if lst_spec_data_objs:
        if doplot:
            fig_filename = (
                "{}_{}.pdf".format(
                    fig_prefix, fits_file_hdr["name"]))
        else:
            fig_filename = None
        lst_tuple_rets.append(
            ex_obj.submit(
                proc_spec_data_obj,
                *(lst_spec_data_objs, config_dct, dct_options,
                  fig_filename),
                **{"ccfinit":ccfinit, "doplot":doplot,
                   "dct_priors":dct_priors}
            )
        )
        lst_times.append(time())
        for tuple_rets in lst_tuple_rets:
            outdct, curmodel = tuple_rets.result()
            vers = outdct["vers"]
            del outdct["vers"]  # I don't want to store it in the table
            
            for col in lst_columns_copy:
                outdct[col] = fits_file_hdr[col]

            outdct["SUCCESS"] = outdct["vel_warn"] == 0
            lst_outdf.append(outdct)

            # This will store best-fit model data
            lst_arr_fluxes_phoenix_syn_stellar_spec_mod = curmodel
        lst_times.append(time())
        outtab = Table(lst_outdf)

        lst_outmod_hdus = [
            PrimaryHDU(
                header=get_prim_header(dct_versions=vers,
                                       config_dct=config_dct,
                                       cmdline=cmdline,
                                       spectrum_header=fits_file_hdr,
                                       dct_priors=dct_priors)
            )
        ]

        lst_outmod_hdus.append(ImageHDU(arr_wavs, name="wavelength"))
        lst_outmod_hdus.append(
            ImageHDU(np.vstack(curmodel), name="model"))

        dct_column_desc = get_column_desc()

        lst_outtab_hdus = [
            PrimaryHDU(
                header=get_prim_header(
                    dct_versions=vers, config_dct=config_dct,
                    cmdline=cmdline, dct_priors=dct_priors)
            ),
            comment_filler(
                BinTableHDU(outtab, name="vel_tab"),
                dct_column_desc)
        ]
        lst_times.append(time())

        write_hdulist(mod_ofilename, HDUList(lst_outmod_hdus))
        write_hdulist(tab_ofilename, HDUList(lst_outtab_hdus))
        lst_times.append(time())
        debug(
            str.format(
                "Global timing: {}", (np.diff(np.array(lst_times)), ))
        )
        return 1
    else:
        warning(
            "Giving up on fitting spectra for row {}".format(
                name_file_w_spec_data)
        )
        
    
def proc_spec_wrapper(*tuple_args, **dct_kwargs):
    status = ProcessStatus.SUCCESS
    status_file = dct_kwargs["proc_status_file"]
    throw_exceptions = dct_kwargs["throw_exceptions"]
    del dct_kwargs["proc_status_file"]
    del dct_kwargs["throw_exceptions"]
    no_fit = 0
    time0 = time()
    try:
        no_fit = proc_file_w_spec(*tuple_args, **dct_kwargs)
    except:  # noqa F841
        status = ProcessStatus.FAILURE
        exception(
            "failed with these arguments{}{}".format(
                str(tuple_args), str(dct_kwargs))
        )
        pid = getpid()
        logfilename = (
            "crash_{:d}_{}.log".format(pid, ctime().replace(" ", ""))
        )
        with open(logfilename, "w") as file_obj:
            print("failed with these arguments", tuple_args, dct_kwargs,
                  file=file_obj)
            print_exc(file=file_obj)
        # I decided not to just fail, proceed instead after
        # writing a bug report
        if throw_exceptions:
            raise
    finally:
        time1 = time()
        if status_file:
            if no_fit < 0:
                status = ProcessStatus.FAILURE
                no_fit = 0
            upd_proc_status_file(status_file, tuple_args[0], status,
                                 no_fit, time1 - time0, start=False)


proc_spec_wrapper.__doc__ = proc_file_w_spec.__doc__


def proc_many(files, output_dir, output_tab_prefix, output_mod_prefix,
              config_name, ccf_continuum_normalize=True,
              config_filename=None, no_threads=1, proc_status_file=None,
              subdirs=True, fig_dir=None, fig_prefix=None,
              skipexisting=False, wav0=None, wav1=None,
              throw_exceptions=None, npoly=None, doplot=True,
              ccfinit=True, cmdline=None,
              eff_temp_kelvin_normal_prior_means=None,
              eff_temp_kelvin_normal_prior_st_devs=None,
              lg_g_normal_prior_means=None,
              lg_g_normal_prior_st_devs=None,
              fe_h_normal_prior_means=None,
              fe_h_normal_prior_st_devs=None,
              alpha_fe_normal_prior_means=None,
              alpha_fe_normal_prior_st_devs=None):
    """
    Process many spectral files

    Parameters
    ----------
    files : strings
        The files with spectra 
    output_dir : string
        Output directory for the tables
    output_tab_prefix : string
        Prefix of output table files
    output_mod_prefix : string
        Prefix of output model files
    config_name: string
        The name of the setup
    ccf_continuum_normalize : bool, default True          
        By default normalize by the continuum when doing CCF
    config_filename: string, default None
        The name of the config file
    no_threads : int, default 1
        Number of threads for the fits
    proc_status_file: str
        The filename where we'll put status of the fitting
    subdirs : bool, default True
        Default to creating the subdirectories in the output dir
    fig_dir : string, default None
        The directory where the figures will be stored
    fig_prefix : string, default None
        The prefix of figure filenames
    skipexisting: bool, default False
        If True, do not process anything if output files exist
    wav0 : float, default None
        Wavelength of the lower limit of the considered range across
        dispersion axis of spectrum (should be in same units as unit
        used for wavelengths of dispersion axis)
    wav1 : float, default None
        Wavelength of the upper limit of the considered range across
        dispersion axis of spectrum (should be in same units as unit
        used for wavelengths of dispersion axis)
    throw_exceptions : bool, default None
        If this option is set, the code will not protect against
        exceptions inside RVSpecFit.
    npoly : integer, default None
        The degree of the polynomial used for continuum
    doplot : bool, default True
        Plotting
    ccfinit : bool, default True
         If true (default the starting point will be determined from
         cross-correlation as opposed to brute-force grid search
    cmdline : string, default None
        The command line used in execution of RVSpecFit
    eff_temp_kelvin_normal_prior_means : strings, default None
        Means of normal priors of effective temperature in Kelvin for
        each star
    eff_temp_kelvin_normal_prior_st_devs : strings, default None
        Standard deviations of normal priors of effective temperature in
        Kelvin for each star
    lg_g_normal_prior_means : strings, default None
        Means of normal priors of logarithm of surface gravity for each
        star
    lg_g_normal_prior_st_devs : strings, default None
        Standard deviations of normal priors of logarithm of surface
        gravity for each star
    fe_h_normal_prior_means : strings, default None
        Means of normal priors of [Fe/H] for each star
    fe_h_normal_prior_st_devs : strings, default None
        Standard deviations of normal priors of [Fe/H] for each star
    alpha_fe_normal_prior_means : strings, default None
        Means of normal priors of [α/Fe] for each star
    alpha_fe_normal_prior_st_devs : strings , default None
        Standard deviation of normal prior of [α/Fe] for each star
    """
    override_dct = {"ccf_continuum_normalize":ccf_continuum_normalize}
    config_dct = read_config(config_filename, override_dct)
    assert (config_dct is not None)
    assert ("template_lib" in config_dct)

    if no_threads:
        parallel = True
    else:
        parallel = False
    if proc_status_file:
        upd_proc_status_file(proc_status_file, None, None, None, None,
                             start=True)
    if parallel:
        ex_obj = ProcessPoolExecutor(no_threads)
    else:
        ex_obj = FakeExecutor()
    lst_res = []
    tuple_file_queue_objs = (files, eff_temp_kelvin_normal_prior_means,
                             eff_temp_kelvin_normal_prior_st_devs,
                             lg_g_normal_prior_means,
                             lg_g_normal_prior_st_devs,
                             fe_h_normal_prior_means,
                             fe_h_normal_prior_st_devs,
                             alpha_fe_normal_prior_means,
                             alpha_fe_normal_prior_st_devs)
    for tuple_entry_file_queue_objs in zip(*tuple_file_queue_objs): 
        filename = tuple_entry_file_queue_objs[0].split("/")[-1]
        if subdirs:
            assert (len(tuple_entry_file_queue_objs[0].split("/")) > 2)
            # We need that because we use the last two directories in
            # the path to create output directory structure, i.e., input
            # file a/b/c/d/e/f/g.fits will produce output file in
            # output_prefix/e/f/xxx.fits
            lst_file_dirs = tuple_entry_file_queue_objs[0].split("/")
            folder_path = (
                "{}/{}/{}/".format(output_dir, lst_file_dirs[-3],
                                   lst_file_dirs[-2])
            )
        else:
            folder_path = "{}/".format(output_dir)
        if isdir(folder_path):
            pass
        else:
            makedirs(folder_path)
        debug("Making folder {}".format(folder_path))
        if fig_dir:
            if subdirs:
                fig_path = (
                    "{}/{}/{}/".format(fig_dir, lst_file_dirs[-3],
                                       lst_file_dirs[-2])
                )
            else:
                fig_path = "{}/".format(fig_dir)
            if isdir(fig_path):
                pass
            else:
                makedirs(fig_path)
            cur_fig_prefix = "{}/{}".format(fig_path, fig_prefix)
            debug("Making folder {}".format(fig_path))
        else:
            cur_fig_prefix = None
        if filename[-3:] == ".gz":
            filename0 = filename[:-3]
        else:
            filename0 = filename
        tab_ofilename = (
            "{}{}_{}".format(folder_path, output_tab_prefix, filename0))
        mod_ofilename = (
            "{}{}_{}".format(folder_path, output_mod_prefix, filename0))
            
        dct_priors = {}
        for tuple_normal_prior_pars in [
            (tuple_entry_file_queue_objs[1],
             tuple_entry_file_queue_objs[2], "teff"),
            (tuple_entry_file_queue_objs[3],
             tuple_entry_file_queue_objs[4], "logg"),
            (tuple_entry_file_queue_objs[5],
             tuple_entry_file_queue_objs[6], "feh"),
            (tuple_entry_file_queue_objs[7],
             tuple_entry_file_queue_objs[8], "alphafe")]:
             if all(tuple_normal_prior_pars):
                 dct_priors[tuple_normal_prior_pars[2]] = (
                     float(tuple_normal_prior_pars[0]),
                     float(tuple_normal_prior_pars[1]))

        if (skipexisting and exists(tab_ofilename)
                and exists(mod_ofilename)):
            info(
                "skipping, products already exist {}".format(
                    tuple_entry_file_queue_objs[0])
            )
            if proc_status_file:
                upd_proc_status_file(
                    proc_status_file, tuple_entry_file_queue_objs[0],
                    ProcessStatus.EXISTING, -1, 0)

            continue
        tuple_args = (
            tuple_entry_file_queue_objs[0], config_name, cur_fig_prefix,
            config_dct, mod_ofilename, tab_ofilename)
        dct_kwargs = {
            "proc_status_file":proc_status_file, "wav0":wav0,
            "wav1":wav1, "throw_exceptions":throw_exceptions,
            "npoly":npoly, "doplot":doplot, "ex_obj":ex_obj,
            "ccfinit":ccfinit, "cmdline":cmdline,
            "dct_priors":dct_priors}
        proc_spec_wrapper(*tuple_args, **dct_kwargs)

    if parallel:
        try:
            ex_obj.shutdown(wait=True)
        except KeyboardInterrupt:
            for res in lst_res:
                res.cancel()
            ex_obj.shutdown(wait=False)
            raise

    info("Successfully finished processing")
    
    
def main(args):
    cmdline = " ".join(args)
    arg_parser_obj = ArgumentParser()

    arg_parser_obj.add_argument(
        "--no_threads", help="Number of threads for the fits", type=int,
        default=1)

    arg_parser_obj.add_argument(
        "--config", help="The filename of the configuration file",
        type=str, default=None)

    for arg_name_prefix, arg_name_descr in zip(
        ["input_file", "teff_nprior_mean", "teff_nprior_stdev",
         "lg_g_nprior_mean", "lg_g_nprior_stdev",
         "fe_h_nprior_mean", "fe_h_nprior_stdev", "alpha_nprior_mean",
         "alpha_nprior_stdev"],
         ["spectral files",
          "means of normal priors of effective temperature in Kelviqn"
          " for each star",
          "standard deviations of normal priors of effective"
          " temperature in Kelvin for each star",
          "means of normal priors of logarithm of surface gravity for"
          " each star",
          "standard deviations of normal priors of logarithm of surface"
          " gravity for each star",
          "means of normal priors of [Fe/H] for each star",
          "standard deviations of normal priors of [Fe/H] for each"
          " star",
          "means of normal priors of [/Fe] for each star",
          "standard deviations of normal priors of [/Fe] for each"
          " star"]):
        arg_parser_obj.add_argument(
            "--{}s".format(arg_name_prefix), nargs="*", default=None, 
            help=(
                "Space separated list of {} to process".format(
                    arg_name_descr)
            )
        )
        arg_parser_obj.add_argument(
            "--{}_from".format(arg_name_prefix), default=None,
            help=(
                "Read the list of {} from the text file".format(
                    arg_name_descr)
            )
        )
    arg_parser_obj.add_argument(
        "--queue_file",
        help=(
            "If the input file list is a queue where we delete entries"
            " as soon as we picked up a file"),
        action="store_true", default=False)
    arg_parser_obj.add_argument("--setup", help="The name of the setup",
                                type=str, default=None)

    arg_parser_obj.add_argument(
        "--output_dir", help="Output directory for the tables",
        type=str, default="./", required=False)
    arg_parser_obj.add_argument(
        "--targetid", help="Fit only a given targetid", type=int,
        default=None, required=False)
    arg_parser_obj.add_argument(
        "--targetid_file_from",
        help="Fit only a given targetids from a given file", type=str,
        default=None, required=False)

    arg_parser_obj.add_argument("--npoly", help="npoly", type=int,
                                default=None, required=False)

    arg_parser_obj.add_argument(
        "--output_tab_prefix", help="Prefix of output table files",
        type=str, default="vel_tab", required=False)
    arg_parser_obj.add_argument(
        "--output_mod_prefix", help="Prefix of output model files",
        type=str, default="vel_mod", required=False)
    arg_parser_obj.add_argument(
        "--fig_dir",
        help="Path for the fit figures, i.e. fig_folder/", type=str,
        default="./")
    arg_parser_obj.add_argument(
        "--fig_prefix",
        help="Prefix for the fit figures filename, i.e. im", type=str,
        default="fig", required=False)
    arg_parser_obj.add_argument(
        "--log", help="Log filename", type=str, default=None,
        required=False)
    arg_parser_obj.add_argument(
        "--log_level", help="DEBUG/INFO/WARNING/ERROR/CRITICAL",
        type=str, default="WARNING", required=False)
    arg_parser_obj.add_argument(
        "--param_init",
        help=(
            "How the initial parameters/line-of-sight velocity are"
            " initialized"),
        type=str, default="CCF", required=False)
    arg_parser_obj.add_argument(
        "--proc_status_file",
        help=(
            "The name of the file where I put the names of successfully"
            " processed files"),
        type=str, default=None, required=False)
    arg_parser_obj.add_argument(
        "--overwrite",
        help=(
            "If enabled, the code will overwrite the existing products."
            "Otherwise, it will attempt to update/append"),
        default=None, required=False)
    arg_parser_obj.add_argument(
        "--vers", help="Output the version of the software",
        action="store_true", default=False)

    arg_parser_obj.add_argument(
        "--skipexisting",
        help=(
            "If enabled, the code will completely skip if there are"
            " existing products"),
        action="store_true", default=False)

    arg_parser_obj.add_argument(
        "--doplot", help="Make plots", action="store_true",
        default=False)

    arg_parser_obj.add_argument(
        "--no_ccf_continuum_normalize",
        help="Do not normalize by the continuum when doing CCF",
        dest="ccf_continuum_normalize", action="store_false",
        default=True)
    arg_parser_obj.add_argument(
        "--no_subdirs",
        help="Do not create the subdirectories in the output dir",
        dest="subdirs", action="store_false", default=True)

    arg_parser_obj.add_argument(
        "--throw_exceptions",
        help=(
            "If this option is set, the code will not protect against"
            " exceptions inside rvspecfit"),
            action="store_true", default=False)
            
    for arg_name_suffix, arg_name_suffix_descr in zip(
        [0, 1], ["lower", "upper"]):
        arg_parser_obj.add_argument(
            "--wav{}".format(arg_name_suffix), default=None, type=float,
            help=(
                "".join(
                    ["Wavelength of the {} limit of the".format(
                         arg_name_suffix_descr),
                     " considered range across dispersion axis of"
                     " spectrum (should be in same units as unit used"
                     " for wavelengths of dispersion axis)"]
                )
            )
        )

    arg_parser_obj.set_defaults(ccf_continuum_normalize=True)
    namespace_obj = arg_parser_obj.parse_args(args)

    if namespace_obj.vers:
        print(_version.version)
        exit(0)
        
    log_level = namespace_obj.log_level

    if namespace_obj.log:
        basicConfig(filename=namespace_obj.log, level=log_level)
    else:
        basicConfig(level=log_level)
        
    input_files = namespace_obj.input_files
    input_file_from = namespace_obj.input_file_from
    setup_name = namespace_obj.setup
    output_dir, output_tab_prefix, output_mod_prefix = (
        namespace_obj.output_dir, namespace_obj.output_tab_prefix,
        namespace_obj.output_mod_prefix)
    queue_file = namespace_obj.queue_file
    no_threads = namespace_obj.no_threads
    config_fname = namespace_obj.config
    doplot = namespace_obj.doplot
    targetid_file_from = namespace_obj.targetid_file_from
    targetid = namespace_obj.targetid
    npoly = namespace_obj.npoly
    teff_nprior_means = namespace_obj.teff_nprior_means
    teff_nprior_mean_from = namespace_obj.teff_nprior_mean_from
    teff_nprior_stdevs = namespace_obj.teff_nprior_stdevs
    teff_nprior_stdev_from = namespace_obj.teff_nprior_stdev_from
    lg_g_nprior_means = namespace_obj.lg_g_nprior_means
    lg_g_nprior_mean_from = namespace_obj.lg_g_nprior_mean_from
    lg_g_nprior_stdevs = namespace_obj.lg_g_nprior_stdevs
    lg_g_nprior_stdev_from = namespace_obj.lg_g_nprior_stdev_from
    fe_h_nprior_means = namespace_obj.fe_h_nprior_means
    fe_h_nprior_mean_from = namespace_obj.fe_h_nprior_mean_from
    fe_h_nprior_stdevs = namespace_obj.fe_h_nprior_stdevs
    fe_h_nprior_stdev_from = namespace_obj.fe_h_nprior_stdev_from
    alpha_nprior_means = namespace_obj.alpha_nprior_means
    alpha_nprior_mean_from = namespace_obj.alpha_nprior_mean_from
    alpha_nprior_stdevs = namespace_obj.alpha_nprior_stdevs
    alpha_nprior_stdev_from = namespace_obj.alpha_nprior_stdev_from
    if namespace_obj.param_init == "CCF":
        ccfinit = True
    elif namespace_obj.param_init == "bruteforce":
        ccfinit = False
    else:
        raise ValueError(
            "Unknown param_init value; only known ones are CCF and"
            " bruteforce")
    ccf_continuum_normalize = namespace_obj.ccf_continuum_normalize
    
    if input_files and input_file_from:
        raise RuntimeError(
            "".join(
                ["You can only specify {} OR {} options but".format(
                     input_files, input_file_from),
                 " not both of them simulatenously"]
            )
        )
    elif not input_file_from and not input_files:
        arg_parser_obj.print_help()
        raise RuntimeError(
            "You need to specify the spectra you want to fit")
    if input_files == []:
        input_files = None
    elif not input_files:
        input_files = None
    files = FileQueue(file_list=input_files, file_from=input_file_from,
                      queue=queue_file)
    dct_ilsts = {}
    dct_ofile_queue_objs = {}
    for lst_arg_name, file_arg_name, oname in zip(
        ["teff_nprior_means", "teff_nprior_stdevs", "lg_g_nprior_means",
         "lg_g_nprior_stdevs", "fe_h_nprior_means",
         "fe_h_nprior_stdevs", "alpha_nprior_means",
         "alpha_nprior_stdevs"],
        ["teff_nprior_mean_from", "teff_nprior_stdev_from",
         "lg_g_nprior_mean_from", "lg_g_nprior_stdev_from",
         "fe_h_nprior_mean_from", "fe_h_nprior_stdev_from",
         "alpha_nprior_mean_from", "alpha_nprior_stdev_from"],
        ["teff_nprior_means", "teff_nprior_stdevs", "lg_g_nprior_means",
         "lg_g_nprior_stdevs", "fe_h_nprior_means",
         "fe_h_nprior_stdevs", "alpha_nprior_means",
         "alpha_nprior_stdevs"]):
        if locals()[lst_arg_name] and locals()[file_arg_name]:
            raise RuntimeError(
                "".join(
                    ["You can only specify {} OR {} options but".format(
                         lst_arg_name, file_arg_name),
                     " not both of them simulatenously"]
                )
            )
        if locals()[lst_arg_name] == []:
            dct_ilsts[lst_arg_name] = None
        elif not locals()[lst_arg_name]:
            dct_ilsts[lst_arg_name] = None
        if locals()[lst_arg_name] or locals()[file_arg_name]:
            dct_ofile_queue_objs[oname] = FileQueue(
                file_list=dct_ilsts[lst_arg_name],
                file_from=locals()[file_arg_name], queue=queue_file)
        else:
            dct_ofile_queue_objs[oname] = len(files.file_list) * [None]
    fit_targetid = None
    if targetid_file_from and targetid:
        raise RuntimeError(
            "You can only specify targetid or targetid_file_from"
            " options")
    elif targetid_file_from:
        fit_targetid = []
        with open(targetid_file_from, "r") as file_obj:
            for curl in file_obj:
                fit_targetid.append(int(curl.rstrip()))
        fit_targetid = np.unique(fit_targetid)
    elif targetid:
        fit_targetid = np.unique([targetid])
    else:
        pass
    if doplot:
        fig_dir = namespace_obj.fig_dir
    else:
        fig_dir = None

    if namespace_obj.overwrite:
        warning("overwrite keyword is meaningless now")
    proc_many(
        files, output_dir, output_tab_prefix, output_mod_prefix,
        setup_name, ccf_continuum_normalize=ccf_continuum_normalize,
        config_filename=config_fname, no_threads=no_threads,
        proc_status_file=namespace_obj.proc_status_file,
        subdirs=namespace_obj.subdirs, fig_dir=fig_dir,
        fig_prefix=namespace_obj.fig_prefix,
        skipexisting=namespace_obj.skipexisting,
        wav0=namespace_obj.wav0, wav1=namespace_obj.wav1,
        throw_exceptions=namespace_obj.throw_exceptions, npoly=npoly,
        doplot=doplot, ccfinit=ccfinit,
        eff_temp_kelvin_normal_prior_means=(
            dct_ofile_queue_objs["teff_nprior_means"]),
        eff_temp_kelvin_normal_prior_st_devs=(
            dct_ofile_queue_objs["teff_nprior_stdevs"]),
        lg_g_normal_prior_means=(
            dct_ofile_queue_objs["lg_g_nprior_means"]),
        lg_g_normal_prior_st_devs=(
            dct_ofile_queue_objs["lg_g_nprior_stdevs"]),
        fe_h_normal_prior_means=(
            dct_ofile_queue_objs["fe_h_nprior_means"]),
        fe_h_normal_prior_st_devs=(
            dct_ofile_queue_objs["fe_h_nprior_stdevs"]),
        alpha_fe_normal_prior_means=(
            dct_ofile_queue_objs["alpha_nprior_means"]),
        alpha_fe_normal_prior_st_devs=(
            dct_ofile_queue_objs["alpha_nprior_stdevs"])
    )


if __name__ == "__main__":
    main(argv[1:])

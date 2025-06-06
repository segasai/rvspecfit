import os

os.environ['OMP_NUM_THREADS'] = '1'

# The noqa are to prevent warnings due to imports being not at the top
import warnings  # noqa: E402
import sys  # noqa: E402
import re  # noqa: E402
import argparse  # noqa: E402
import time  # noqa: E402
import itertools  # noqa: E402
import traceback  # noqa: E402
import functools  # noqa: E402
import operator  # noqa: E402
import logging  # noqa: E402
import enum  # noqa: E402
from importlib.metadata import version as get_package_version  # noqa: E402
import concurrent.futures  # noqa: E402
import astropy.table as atpy  # noqa: E402
import astropy.io.fits as pyfits  # noqa: E402
import astropy.units as auni  # noqa: E402
import numpy as np  # noqa: E402
import scipy.stats  # noqa: E402
import scipy.sparse  # noqa: E402
import rvspecfit  # noqa: E402
from rvspecfit import fitter_ccf, vel_fit, spec_fit, utils, \
    spec_inter  # noqa: E402


class ProcessStatus(enum.Enum):
    SUCCESS = 0
    FAILURE = 1
    EXISTING = 2

    def __str__(self):
        return self.name


class GlobalConfig:
    table_prefix = 'rvtab'
    model_prefix = 'rvmod'


DEPEND_PACKAGES = [
    'numpy', 'astropy', 'matplotlib', 'rvspecfit', 'pandas', 'scipy', 'yaml',
    'numdifftools', 'h5py', 'torch', 'cffi', 'desitarget'
]

bitmasks = {
    'CHISQ_WARN': 1,  # delta chi-square vs continuum is too larger
    'RV_WARN': 2,  # rv is to close to the edge
    'RVERR_WARN': 4,  # RV error is too large
    'PARAM_WARN': 8,  # parameters are too close to the edge
    'VSINI_WARN': 16,  # vsini is too large
    'BAD_SPECTRUM': 32,  # some issue with the spectrum
    'BAD_HESSIAN': 64  # issue with the hessian matrix
}


def update_process_status_file(status_fname,
                               processed_file,
                               status,
                               nobjects,
                               time_sec,
                               start=False):
    if start:
        with open(status_fname, 'w') as fp:
            pass
        if processed_file is None:
            return
    with open(status_fname, 'a') as fp:
        print(f'{processed_file} {status} {nobjects} {time_sec:.2f}', file=fp)
    return


def get_dep_versions():
    """
    Get Packages versions
    """
    # Ideally you need to check that the list here matches the requirements.txt
    ret = {}
    for curp in DEPEND_PACKAGES:
        try:
            ret[curp] = get_package_version(curp)
        except ImportError:  # this should not really be happening
            # but can, and is not worth bailing out
            pass
    ret['python'] = str.split(sys.version, ' ')[0]
    return ret


def get_zbest_fname(fname):
    """ Get the zbest file for a given spectrum """
    paths = fname.split('/')
    fname_end = paths[-1]
    if fname_end[-3:] == '.gz':
        fname_end = fname_end[:-3]
    not_found = (None, None)  # we return this if we are unsuccessful
    zbest_prefixes = ['redrock-', 'zbest-']
    extensions = ['REDSHIFTS', 'ZBEST']
    file_prefixes = ['coadd-', 'spectra-']
    # I know how to deal with 'coadd-' or 'spectra-' data only
    for curpref in file_prefixes:
        if fname_end[:len(curpref)] == curpref:
            break
    else:
        return not_found
    # try two types of zbest files redrock or zbest (used in the past)
    for cur_zpref, cur_ext in zip(zbest_prefixes, extensions):
        f1 = fname_end.replace(curpref, cur_zpref)
        for postf in ['', '.gz']:
            zbest_path = '/'.join(paths[:-1] + [f1]) + postf
            if os.path.exists(zbest_path):
                return zbest_path, cur_ext
    return not_found


def get_prim_header(versions={},
                    config=None,
                    cmdline=None,
                    spectrum_header=None,
                    zbest_path=None):
    """ Return the Primary HDU with various info in the header

    """
    header = pyfits.Header()
    for i, (k, v) in enumerate(get_dep_versions().items()):
        header['DEPNAM%02d' % i] = (k, 'Software')
        header['DEPVER%02d' % i] = (v, 'Version')
    for i, (k, v) in enumerate(versions.items()):
        header['TMPLCON%d' % i] = (k, 'Spec arm config name')
        header['TMPLREV%d' % i] = (v['revision'], 'Spec template revision')
        header['TMPLSVR%d' % i] = (v['creation_soft_version'],
                                   'Spec template soft version')
    if config is not None:
        header['RVS_CONF'] = config['config_file_path']
    if cmdline is not None:
        header['RVS_CMD'] = cmdline
    if zbest_path is not None:
        if len(zbest_path) < 80:
            # this is the hack to deal with the 'card is too long comment will
            # be truncated
            zbest_path = zbest_path + (' ' * (80 - len(zbest_path)))
    header['RR_FILE'] = (zbest_path or '', 'Redrock redshift file')
    # keywords to copy from the header of the spectrum
    copy_keys = [
        'SPGRP', 'SPGRPVAL', 'TILEID', 'SPECTRO', 'PETAL', 'NIGHT', 'EXPID',
        'HPXPIXEL', 'HPXNSIDE', 'HPXNEST'
    ]

    if spectrum_header is not None:
        for key in copy_keys:
            if key in spectrum_header:
                header[key] = spectrum_header[key]
    return header


def make_plot(specdata, yfit, title, fig_fname):
    """
    Make a plot with the spectra and fits

    Parameters
    ----------

    specdata: SpecData object
        The object with specdata
    yfit: list fo numpy arrays
        The list of fit models
    title: string
        The figure title
    fig_fname: string
        The filename of the figure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    alpha = 0.8
    line_width = 0.8
    dpi = 150
    plt.clf()
    ndat = len(specdata)
    figsize = (10, 3 * ndat)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for i in range(ndat):
        fig.add_subplot(ndat, 1, i + 1)
        curspec = specdata[i].spec
        perc = 0.2
        xind = specdata[i].badmask
        ymin, ymax = [
            scipy.stats.scoreatpercentile(curspec[~xind], _)
            for _ in [perc, 100 - perc]
        ]
        plt.plot(specdata[i].lam, specdata[i].spec, 'k-', linewidth=line_width)
        plt.plot(specdata[i].lam[xind],
                 specdata[i].spec[xind],
                 'b.',
                 linewidth=line_width)
        plt.fill_between(specdata[i].lam,
                         specdata[i].spec - specdata[i].espec,
                         specdata[i].spec + specdata[i].espec,
                         color='grey',
                         alpha=0.1,
                         zorder=10)
        plt.plot(specdata[i].lam[xind],
                 specdata[i].spec[xind],
                 'b.',
                 linewidth=line_width)
        plt.plot(specdata[i].lam,
                 yfit[i],
                 'r-',
                 alpha=alpha,
                 linewidth=line_width)
        plt.ylim(ymin, ymax)
        if i == 0:
            plt.title(title)
        if i == ndat - 1:
            plt.xlabel(r'$\lambda$ [$\AA$]')
    plt.tight_layout()
    plt.savefig(fig_fname)
    plt.close(fig)


def valid_file(FP):
    """
    Check if all required extensions are present if yes return true
    """
    extnames = [_.name for _ in FP]

    names0 = []
    arms = ['B', 'R', 'Z']
    prefs = 'WAVELENGTH', 'FLUX', 'IVAR', 'MASK'
    reqnames = names0 + [
        '%s_%s' % (_, __) for _, __ in itertools.product(arms, prefs)
    ]
    reqnames = reqnames + ['FIBERMAP']
    missing = []
    for curn in reqnames:
        if curn not in extnames:
            missing.append(curn)
    if len(missing) != 0:
        logging.warning('Extensions %s are missing' % (','.join(missing)))
        return False
    return True


def proc_onespec(
    specdata,
    setups,
    config,
    options,
    resolution_matrix=None,
    fig_fname='fig.png',
    ccf_init=True,
    doplot=True,
):
    """Process one single Specdata object

    Parameters
    ----------
    specdata: list
        List of Specdata objects to be fit
    setups: list
        List of strings of spectroscopic configurations
    options: dict
        Configuration options
    fig_fname: str
        Filename for the plot
    doplot: bool
        Do plotting or not
    ccf_init: bool
        If true, the starting point for the fit will be determined
        through CCF rather than bruteforce grid search

    Returns
    -------
    outdict: dict
        Dictionary with fit measurements
    yfit: list
        List of best-fit models
    """
    outdict = {}
    chisqs = {}
    chisqs_c = {}
    t1 = time.time()
    if ccf_init:
        res = fitter_ccf.fit(specdata, config)
        paramDict0 = res['best_par']
        vrad_ccf = res['best_vel']
    else:
        res = vel_fit.firstguess(specdata, config=config, options=options)
        res['best_vsini'] = res.get('vsini')
        paramDict0 = res
    t2 = time.time()
    fixParam = []

    if res['best_vsini'] is not None:
        paramDict0['vsini'] = np.clip(res['best_vsini'], config['min_vsini'],
                                      config['max_vsini'])

    fit_res = vel_fit.process(
        specdata,
        paramDict0,
        fixParam=fixParam,
        config=config,
        options=options,
    )
    t3 = time.time()
    chisq_cont_array = spec_fit.get_chisq_continuum(
        specdata, options=options)['chisq_array']
    t4 = time.time()
    outdict = dict(VRAD=fit_res['vel'] * auni.km / auni.s,
                   VRAD_ERR=fit_res['vel_err'] * auni.km / auni.s,
                   VRAD_SKEW=fit_res['vel_skewness'],
                   VRAD_KURT=fit_res['vel_kurtosis'],
                   VSINI=fit_res['vsini'] * auni.km / auni.s)

    # values are name and unit
    name_mappings = {
        'logg': ('LOGG', 1),
        'teff': ('TEFF', auni.K),
        'feh': ('FEH', 1),
        'alpha': ('ALPHAFE', 1)
    }
    for name1, (name2, unit) in name_mappings.items():
        outdict[name2] = fit_res['param'][name1] * unit
        outdict[name2 + '_ERR'] = fit_res['param_err'][name1] * unit

    npixels = {}
    for i, curd in enumerate(specdata):
        if curd.name not in chisqs:
            chisqs[curd.name] = 0
            chisqs_c[curd.name] = 0
            npixels[curd.name] = 0
        chisqs[curd.name] += fit_res['chisq_array'][i]
        chisqs_c[curd.name] += chisq_cont_array[i]
        npixels[curd.name] += fit_res['npix_array'][i]

    outdict['CHISQ_TOT'] = sum(chisqs.values())
    outdict['CHISQ_C_TOT'] = sum(chisqs_c.values())
    outdict['NPIX_TOT'] = sum(npixels.values())

    for s in chisqs.keys():
        outdict['CHISQ_%s' % s.replace('desi_', '').upper()] = chisqs[s]
        outdict['CHISQ_C_%s' % s.replace('desi_', '').upper()] = float(
            chisqs_c[s])
    if ccf_init:
        outdict['VRAD_CCF'] = vrad_ccf * auni.km / auni.s

    outdict['RVS_WARN'] = get_rvs_warn(fit_res, outdict, config)

    if doplot:
        title = ('logg=%.1f teff=%.1f [Fe/H]=%.1f ' +
                 '[alpha/Fe]=%.1f Vrad=%.1f+/-%.1f vsini=%.1f') % (
                     fit_res['param']['logg'], fit_res['param']['teff'],
                     fit_res['param'].get('feh')
                     or 0, fit_res['param'].get('alpha') or 0, fit_res['vel'],
                     fit_res['vel_err'], fit_res.get('vsini') or 0)
        if len(specdata) > len(setups):
            for i in range(len(specdata) // len(setups)):
                sl = slice(i * len(setups), (i + 1) * len(setups))
                make_plot(specdata[sl], fit_res['yfit'][sl], title,
                          fig_fname.replace('.png', '_%d.png' % i))
        else:
            make_plot(specdata, fit_res['yfit'], title, fig_fname)
    versions = {}
    for i, (k, v) in enumerate(spec_inter.interp_cache.interps.items()):
        versions[k] = dict(revision=v.revision,
                           creation_soft_version=v.creation_soft_version)
    outdict['versions'] = versions
    logging.debug('Timing: %.4f %.4f %.4f' % (t2 - t1, t3 - t2, t4 - t3))
    return outdict, fit_res['yfit']


def get_rvs_warn(fit_res, outdict, config):
    chisq_thresh = 50
    # if the delta-chisq between continuum is smaller than this we
    # set a warning flag
    # if feh is close to the edge
    feh_thresh = 0.01
    feh_edges = [-4, 1]

    teff_thresh = 10
    teff_edges = [2300, 15000]

    logg_thresh = 0.01
    logg_edges = [-.5, 6.5]

    # if we are within this threshold of the RV boundary we set another
    # warning
    rvedge_thresh = 5

    rverr_thresh = 100
    # If the error is larger than this we warn

    vsini_thresh = 20

    rvs_warn = 0

    dchisq = outdict['CHISQ_C_TOT'] - outdict['CHISQ_TOT']  # should be >0

    if (dchisq < chisq_thresh):
        rvs_warn |= bitmasks['CHISQ_WARN']
    kms = auni.km / auni.s
    cur_vrad = outdict['VRAD'].to_value(kms)
    if _bad_edge_check(cur_vrad, [config['min_vel'], config['max_vel']],
                       rvedge_thresh):
        rvs_warn |= bitmasks['RV_WARN']
    if outdict['VSINI'].to_value(kms) > vsini_thresh:
        rvs_warn |= bitmasks['VSINI_WARN']
    if (outdict['VRAD_ERR'].to_value(kms) > rverr_thresh):
        rvs_warn |= bitmasks['RVERR_WARN']

    if fit_res['bad_hessian']:
        rvs_warn |= bitmasks['BAD_HESSIAN']
    # Here we check if the parameter is within the threshold of the edge or
    # beyond the edge.
    parameter_limits = [['teff', teff_edges, teff_thresh],
                        ['feh', feh_edges, feh_thresh],
                        ['logg', logg_edges, logg_thresh]]
    for cur_param, cur_edges, cur_thresh in parameter_limits:
        if _bad_edge_check(fit_res['param'][cur_param], cur_edges, cur_thresh):
            rvs_warn |= bitmasks['PARAM_WARN']
    return rvs_warn


def _bad_edge_check(value, edges, threshold):
    """
    Return true if value is outside the edge or within the
    threshold to the edge
    """
    bad = False
    if ((value < edges[0] + threshold) or (value > edges[1] - threshold)):
        bad = True
    return bad


def get_sns(data, ivars, masks):
    """
    Return the vector of S/Ns
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xind = (ivars <= 0) | (masks > 0)
        xsn = data * np.sqrt(ivars)
        xsn[xind] = np.nan
        sns = np.nanmedian(xsn, axis=1)
        sns[~np.isfinite(sns)] = -1e9  # set problematic spectrum sn to
        # a very negative number
    return sns


def read_data(FP, setups):
    """ Read the data file

    Parameters
    ----------
    fname: str
        Filename
    setups: list
        List of spectral configurations (i.e. ['b', 'r', 'z']

    Returns
    -------
    fluxes: ndarray
        2d array of fluxes
    ivars: ndarray
        2d array of inverse variances
    masks: ndarray
        2d array of masks
    waves: ndarray
        1d array of wavelengths

    """
    fluxes = {}
    ivars = {}
    waves = {}
    masks = {}
    resolutions = {}
    for s in setups:
        fluxes[s] = FP['%s_FLUX' % s.upper()].data
        ivars[s] = FP['%s_IVAR' % s.upper()].data
        masks[s] = FP['%s_MASK' % s.upper()].data
        waves[s] = FP['%s_WAVELENGTH' % s.upper()].data
        resolutions[s] = FP['%s_RESOLUTION' % s.upper()].data
    return fluxes, ivars, masks, waves, resolutions


def filter_fibermap(fibermapT, DT, objtypes=None):
    # compute the subset based on TARGET types
    types_subset = np.ones(len(fibermapT), dtype=bool)
    re_types = [re.compile(_) for _ in objtypes]
    for i, currow in enumerate(fibermapT):
        col_list, mask_list, _ = DT.main_cmx_or_sv(currow, scnd=True)
        # collist will be column list like
        # DESI_TARGET, BGS_TARGET, MWS_TARGET

        # extract the DESI_TARGET part
        colname = col_list[0]
        mask = mask_list[0]

        # all the possible types here
        objtypnames = list(mask.names())
        objs = []
        # check which names match our regular expression
        for curo in objtypnames:
            for r in re_types:
                if r.match(curo) is not None:
                    objs.append(curo)
        # obtain integer values for each object type that matched
        # our RE and bitwise OR them
        bitmask = functools.reduce(operator.or_, [mask.mask(_) for _ in objs])
        # check if the given row has any hits in the bitmask
        types_subset[i] = (currow[colname] & bitmask) > 0
    return types_subset


def select_fibers_to_fit(fibermap,
                         sns,
                         zbest_path=None,
                         zbest_ext=None,
                         minsn=None,
                         objtypes=None,
                         expid_range=None,
                         fit_targetid=None,
                         zbest_select=False,
                         zbest_include=False):
    """ Identify fibers to fit

    Parameters
    ----------
    fibermap: Table
        Fibermap table object
    sns: dict of numpy arrays
        Array of S/Ns
    minsn: float
        Threshold S/N
    objtypes: list of regular expressions
           of DESI_TARGETs ['MWS_ANY,'STD_*'] or None
    expid_range: list
        The range of EXPID to consider
    fit_targetid: list of ints
        Fit only specific TARGETIDs

    Returns
    -------
    ret: bool numpy array
        Array with True for selected spectra
    rr_z: numpy array
        Array of redshifts
    rr_spectype: numpy array
        Array of redshift types
    rr_subtype: numpy array
        Array of redshift subtypes
    """
    zbest_maxvel = 1500  # maximum velocity to consider a star
    zbest_type = 'STAR'

    try:
        import desitarget.targets as DT
    except ImportError:
        DT = None

    subset = np.ones(len(fibermap), dtype=bool)

    # Always apply EXPID range
    if expid_range is not None:
        mine, maxe = expid_range
        if mine is None:
            mine = -1
        if maxe is None:
            maxe = np.inf
    if "EXPID" in fibermap.columns.names:
        subset = subset & (fibermap["EXPID"] > mine) & (fibermap['EXPID']
                                                        <= maxe)
    # ONLY select good fiberstatus ones
    good_fiberstatus = [0, 8]
    # 8 is Positioner has restricted reach (but might still be on valid target)
    # https://desidatamodel.readthedocs.io/en/latest/bitmasks.html
    if 'FIBERSTATUS' in fibermap.columns.names:
        subset = subset & np.isin(fibermap['FIBERSTATUS'], good_fiberstatus)
    elif 'COADD_FIBERSTATUS' in fibermap.columns.names:
        subset = subset & np.isin(fibermap['COADD_FIBERSTATUS'],
                                  good_fiberstatus)

    # Exclude skys and bad but include anything else
    subset = subset & (fibermap['OBJTYPE'] != 'SKY') & (fibermap['OBJTYPE']
                                                        != 'BAD')

    # Always apply TARGETID selection if provided
    if fit_targetid is not None:
        subset = subset & np.isin(fibermap['TARGETID'], fit_targetid)

    # Always apply SN cut if provided
    if minsn is not None:
        maxsn = np.max(np.array(list(sns.values())), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            subset = subset & (maxsn > minsn)

    # compute the subset based on TARGET types
    fibermapT = atpy.Table(fibermap)
    if DT is not None and objtypes is not None:
        selecting_by_type = True
        types_subset = filter_fibermap(fibermapT, DT, objtypes=objtypes)
    else:
        types_subset = np.zeros(len(fibermap), dtype=bool)
        selecting_by_type = False
    # if we are not doing a selection the mask is filled with false

    # select objects based on redrock velocity or type
    selecting_by_zbest = False
    rr_z = None
    rr_spectype = None
    rr_subtype = None
    if zbest_select or zbest_include:
        if zbest_path is None:
            logging.warning(
                'zbest selection requested, but the zbest file not found')
        else:
            if zbest_select:
                selecting_by_zbest = True
            logging.info('Using redshift file %s', zbest_path)
            zb = atpy.Table().read(zbest_path,
                                   format='fits',
                                   hdu=zbest_ext,
                                   mask_invalid=False)
            rr_spectype = zb['SPECTYPE']
            rr_subtype = zb['SUBTYPE']
            rr_z = zb['Z']
            zbest_subset = ((rr_spectype == zbest_type) |
                            ((np.abs(rr_z)) < zbest_maxvel / 3e5))
            if len(zb) == len(subset):
                # row match
                assert np.all(zb['TARGETID'] == fibermap['TARGETID'])
            else:
                # match by id
                # useful when fitting spectra- file using coadd rr file
                xmap = dict(
                    zip(zb['TARGETID'], zip(rr_z, rr_spectype, rr_subtype)))
                zbest_subset = np.isin(fibermap['TARGETID'],
                                       zb['TARGETID'][zbest_subset])
                rr_z = np.zeros(len(fibermap), dtype=zb['Z'].dtype) + np.nan
                rr_spectype = np.ma.zeros(len(fibermap),
                                          dtype=rr_spectype.dtype)
                rr_subtype = np.ma.zeros(len(fibermap), dtype=rr_subtype.dtype)
                for i in range(len(fibermap)):
                    cur_tid = fibermap['TARGETID'][i]
                    if cur_tid in xmap:
                        rr_z[i], rr_spectype[i], rr_subtype[i] = xmap[cur_tid]
    if not selecting_by_zbest:
        zbest_subset = np.zeros(len(fibermap), dtype=bool)
    # if we are not doing a selection the mask is filled with false

    if selecting_by_zbest or selecting_by_type:
        # We select either based on type or zbest
        subset = subset & (zbest_subset | types_subset)
    return subset, rr_z, rr_spectype, rr_subtype


def resolution_mat_torows(mat):
    w = mat.shape[0]
    w2 = w // 2
    return np.array([np.roll(mat[_], _ - w2) for _ in range(w)])[::-1]


def resolution_mat_tocolumns(mat):
    w = mat.shape[0]
    w2 = w // 2
    return np.array([np.roll(mat[::-1][_], w2 - _) for _ in range(w)])


def deconvolve_resolution_matrix(mat0,
                                 sigma0_angstrom=0.5,
                                 pix_size_angstrom=0.8):
    width, npix = mat0.shape
    sig_pix = sigma0_angstrom / pix_size_angstrom
    xs = np.arange(width)
    gau_mat = np.array([
        1. / np.sqrt(2 * np.pi) / sig_pix *
        np.exp(-0.5 * ((xs - i) / sig_pix)**2) for i in range(len(xs))
    ])
    w2 = width // 2
    mat_rows = resolution_mat_torows(mat0)
    # Now this stores rows of the banded matrix rather than columns
    for i in range(w2):
        mat_rows[:w2 - i - 1, i] = 0
        # mat_rows[:, i] = mat_rows[:, i] / mat_rows[:, i].sum()
        j = npix - 1 - i
        mat_rows[w2 + 1 + i:, j] = 0
        # mat_rows[:, j] = mat_rows[:, j] / mat_rows[:, j].sum()

    mat_rows1 = scipy.linalg.solve(gau_mat, mat_rows)
    # shift backward
    mat = resolution_mat_tocolumns(mat_rows1)
    # this a hack no deconvolution of edges
    # mat[:, :w2] = mat0[:, :w2]
    # mat[:, -w2:] = mat0[:, -w2:]
    return mat


def construct_resolution_sparse_matrix(mat,
                                       pix_size_angstrom=None,
                                       sigma0_angstrom=None):
    width, npix = mat.shape
    w2 = width // 2
    mat = mat.copy()
    deconvolve = True
    if deconvolve:
        mat = deconvolve_resolution_matrix(mat,
                                           pix_size_angstrom=pix_size_angstrom,
                                           sigma0_angstrom=sigma0_angstrom)
    # this is because the resolution matrix is incorrect in the edges
    mat_rows = resolution_mat_torows(mat)
    mult = np.median(mat_rows.sum(axis=0))
    if mult == 0:
        mult = 1
    for i in range(w2):
        N1 = mat_rows[w2 - i:, i].sum()
        mat_rows[:, i] = mat_rows[:, i] / (N1 + (N1 == 0)) * mult
        j = npix - 1 - i
        N2 = mat_rows[:w2 + 1 + i, j].sum()
        mat_rows[:, j] = mat_rows[:, j] / (N2 + (N2 == 0)) * mult
    mat = resolution_mat_tocolumns(mat_rows)
    M = scipy.sparse.dia_matrix((mat, np.arange(w2, -w2 - 1, -1)),
                                (npix, npix))
    return M


def interpolate_bad_regions(spec, mask):
    """
    Interpolate bad regions
    """
    xind = np.nonzero(mask)[0]
    if len(xind) == 0:
        return spec
    elif len(xind) == len(spec):
        return spec
    edges = np.nonzero(np.diff(xind, prepend=-10) > 1)[0]
    n_edges = len(edges)
    spec1 = spec * 1
    for i in range(n_edges):
        if i == n_edges - 1:
            rh = xind[-1]
        else:
            rh = xind[edges[i + 1] - 1]
        lh = xind[edges[i]]
        # lh rh inclusive
        if lh == 0:
            spec1[:rh + 1] = spec[rh + 1]
        elif rh == len(spec) - 1:
            spec1[lh:] = spec[lh - 1]
        else:
            spec1[lh:rh + 1] = np.interp(np.arange(lh,
                                                   rh + 1), [lh - 1, rh + 1],
                                         [spec[lh - 1], spec[rh + 1]])
    return spec1


def get_specdata(waves,
                 fluxes,
                 ivars,
                 masks,
                 resolutions,
                 seqid,
                 setups,
                 use_resolution_matrix=False,
                 mask_dicroic=True,
                 lsf_sigma0_angstrom=None):
    """ Return the list of SpecDatas for one single object

    Parameters
    ----------

    waves: ndarray
        1d wavelength array
    fluxes: ndarray
        2d flux array
    ivars: ndarray
        2d array of inverse variances
    masks: ndarray
        2d array of masks
    seqid: int
        Which spectral row to extract
    setups: list
        List of configurations

    Returns
    -------
    ret: list
        List of specfit.SpecData objects or None if failed

    """
    large_error = 1000  # This used to set the bad error in masked pixels
    minerr_frac = 0.3
    # if the error is smaller than this times median error
    # clamp the uncertainty
    sds = []

    for s in setups:
        spec = fluxes[s][seqid] * 1
        curivars = ivars[s][seqid] * 1
        badmask = (masks[s][seqid] > 0)
        medspec = np.nanmedian(spec)
        if badmask.all():
            # skip bad data
            continue
        if medspec == 0:
            medspec = np.nanmedian(spec[(spec > 0) & (~badmask)])
            if not np.isfinite(medspec):
                medspec = np.nanmedian(np.abs(spec))
        if not np.isfinite(medspec) or medspec == 0:
            # bail out the spectrum is insane
            continue
        baddat = ~np.isfinite(spec + curivars)
        if mask_dicroic:
            dicroicmask = (waves[s] > 4300) & (waves[s] < 4450)
        else:
            dicroicmask = np.zeros(len(waves[s]), dtype=bool)
        baderr = curivars <= 0
        edge_mask = np.zeros(len(spec), dtype=bool)
        if use_resolution_matrix:
            dwave = waves[s][1] - waves[s][0]
            cur_sig0 = lsf_sigma0_angstrom[s]
            cur_resol = spec_fit.ResolMatrix(
                construct_resolution_sparse_matrix(resolutions[s][seqid],
                                                   pix_size_angstrom=dwave,
                                                   sigma0_angstrom=cur_sig0))
            edge_pixels = 5
            edge_mask[:edge_pixels] = True
            edge_mask[-edge_pixels:] = True
            # this is because resolution matrix is corrupted
            # edges
        else:
            cur_resol = None

        badall = baddat | badmask | baderr | dicroicmask | edge_mask
        badall_interp = baddat | badmask | baderr
        # do not interpolate edges or dicroic
        curivars[badall] = 1. / medspec**2 / large_error**2
        spec[:] = interpolate_bad_regions(spec, badall_interp)
        espec = 1. / curivars**.5
        if badall.all():
            logging.warning('The whole spectrum was masked...')
        else:
            goodespec = espec[~badall]
            goodespec_thresh = np.median(goodespec) * minerr_frac
            replace_idx = (espec < goodespec_thresh) & (~badall)
            if replace_idx.sum() / (~badall).sum() > .01:
                logging.warning(
                    'More than 1% of spectra had the uncertainty clamped')
            # logging.debug("Clamped error on %d pixels" % (replace_idx.sum()))
            espec[replace_idx] = goodespec_thresh

        sd = spec_fit.SpecData('desi_%s' % s,
                               waves[s],
                               spec,
                               espec,
                               resolution=cur_resol,
                               badmask=badall)

        sds.append(sd)
    if len(sds) == 0:
        # no good data
        logging.warning(f'No good data found for fiber {seqid}')
        return None
    return tuple(sds)


def comment_filler(tab, desc):
    """ Fill comments in the FITS header """
    for i, k in enumerate(tab.data.columns):
        comm = desc.get(k.name)
        if comm is None:
            comm = ''
        else:
            comm = comm[1]
        tab.header['TCOMM%d' % (i + 1)] = comm
    return tab


def put_empty_file(fname):
    """ Write a dummy empty file if we didn't process any spectra """
    pyfits.PrimaryHDU(header=get_prim_header()).writeto(fname,
                                                        overwrite=True,
                                                        checksum=True)


def get_column_desc(setups):
    """ Return the list of column descriptions
    when we fitting a given set of configurations
    """
    columnDesc = dict([
        ('VRAD', (np.float32, 'Radial velocity')),
        ('VRAD_ERR', (np.float32, 'Radial velocity error')),
        ('VRAD_SKEW', (np.float32, 'Radial velocity posterior skewness')),
        ('VRAD_KURT', (np.float32, 'Radial velocity posterior kurtosis')),
        ('VSINI', (np.float32, 'Stellar rotation velocity')),
        ('LOGG', (np.float32, 'Log of surface gravity')),
        ('TEFF', (np.float32, 'Effective temperature')),
        ('FEH', (np.float32, '[Fe/H] from template fitting')),
        ('ALPHAFE', (np.float32, '[alpha/Fe] from template fitting')),
        ('LOGG_ERR', (np.float32, 'Log of surface gravity uncertainty')),
        ('TEFF_ERR', (np.float32, 'Effective temperature uncertainty')),
        ('FEH_ERR', (np.float32, '[Fe/H] uncertainty from template fitting')),
        ('ALPHAFE_ERR', (np.float32,
                         '[alpha/Fe] uncertainty from template fitting')),
        ('CHISQ_TOT', (np.float64, 'Total chi-square for all arms')),
        ('NPIX_TOT', (np.float64, 'Total number of unmasked pixels fitted')),
        ('CHISQ_C_TOT',
         (np.float64,
          'Total chi-square for all arms for polynomial only fit')),
        ('VRAD_CCF', (np.float32, 'Initial velocity from cross-correlation')),
        ('TARGETID', (np.int64, 'DESI targetid')),
        ('EXPID', (np.int64, 'DESI exposure id')),
        ('SUCCESS', (bool, "Did we succeed or fail")),
        ('RVS_WARN', (np.int64, "RVSpecFit warning flag")),
        ('RR_Z', (np.float64, 'Redrock redshift')),
        ('RR_SPECTYPE', (str, 'Redrock spectype')),
        ('RR_SUBTYPE', (str, 'Redrock spectroscopic subtype'))
    ])

    for curs in setups:
        curs = curs.upper()
        columnDesc['SN_%s' % curs] = (np.float32,
                                      'Median S/N in the %s arm' % curs)
        columnDesc['CHISQ_%s' % curs] = (np.float64,
                                         'Chi-square in the %s arm' % curs)
        columnDesc['CHISQ_C_%s' % curs] = (
            np.float64,
            'Chi-square in the %s arm after fitting continuum only' % curs)
    return columnDesc


def proc_desi(fname,
              tab_ofname,
              mod_ofname,
              fig_prefix,
              config,
              fit_targetid=None,
              objtypes=None,
              doplot=True,
              minsn=-1e9,
              expid_range=None,
              poolex=None,
              fitarm=None,
              cmdline=None,
              zbest_select=False,
              zbest_include=False,
              use_resolution_matrix=False,
              ccf_init=True,
              npoly=10):
    """
    Process One single file with desi spectra

    Parameters
    -----------
    fname: str
        The filename with the spectra to be fitted
    ofname: str
        The filename where the table with parameters will be stored
    fig_prefix: str
        The prefix where the figures will be stored
    config: Dictionary
        The configuration dictionary
    fit_targetid: int
        The targetid to fit. If none fit all.
    objtypes: list
        The list of DESI_TARGET types
    doplot: bool
        Produce plots
    minsn: real
        The slallest S/N for processing
    expid_range: tuple of ints
        The range of expids to fit
    fitarm: list
         List of strings corresponding to configurations that need to be fit,
         it can be none
    zbest_select: bool
         If true then the zbest file is used to preselect targets
    ccf_init: bool
         If true (default the starting point will be determined from
         crosscorrelation as opposed to bruteforce grid search
    poolex: Executor
         The executor that will run parallel processes
    """

    if npoly is None:
        npoly = 10
    options = {'npoly': npoly}
    timers = []
    timers.append(time.time())
    logging.info('Processing %s' % fname)
    try:
        FP = pyfits.open(fname)
    except OSError:
        logging.error('Cannot read file %s' % (fname))
        return -1
    valid = valid_file(FP)
    if not valid:
        logging.error('Not valid file: %s' % (fname))
        return -1

    setups = ['b', 'r', 'z']
    if fitarm is not None:
        setups = [_ for _ in setups if _ in fitarm]
        assert (len(setups) > 0)

    spectrum_header = FP[0].header
    fibermap = FP['FIBERMAP'].data
    scores = FP['SCORES'].data
    if 'EXP_FIBERMAP' in FP:
        exp_fibermap = FP['EXP_FIBERMAP'].data
    else:
        exp_fibermap = None

    if fit_targetid is not None:
        if not np.isin(fibermap['TARGETID'], fit_targetid).any():
            # skip reading anything if no good TARGETIDs found
            logging.warning('No fibers selected in file %s' % (fname))
            put_empty_file(tab_ofname)
            put_empty_file(mod_ofname)
            FP.close()
            return 0

    fluxes, ivars, masks, waves, resolutions = read_data(FP, setups)
    FP.close()
    # extract SN from the fibermap or compute ourselves
    if 'MEDIAN_CALIB_SNR_' + setups[0].upper() in scores.columns.names:
        sns = dict([(_, scores['MEDIAN_CALIB_SNR_' + _.upper()])
                    for _ in setups])
    elif 'MEDIAN_COADD_SNR_' + setups[0].upper() in scores.columns.names:
        sns = dict([(_, scores['MEDIAN_COADD_SNR_' + _.upper()])
                    for _ in setups])
    elif 'MEDIAN_COADD_FLUX_SNR_' + setups[0].upper() in scores.columns.names:
        sns = dict([(_, scores['MEDIAN_COADD_FLUX_SNR_' + _.upper()])
                    for _ in setups])
    else:
        sns = dict([(_, get_sns(fluxes[_], ivars[_], masks[_]))
                    for _ in setups])

    for _ in setups:
        if len(sns[_]) != len(fibermap):
            logging.warning((
                'WARNING the size of the data in arm %s' +
                'does not match the size of the fibermap; file %s; skipping...'
            ) % (_, fname))
            return -1
    columnDesc = get_column_desc(setups)

    if zbest_select or zbest_include:
        zbest_path, zbest_ext = get_zbest_fname(fname)
    else:
        zbest_path, zbest_ext = None, None

    subset, rr_z, rr_spectype, rr_subtype = select_fibers_to_fit(
        fibermap,
        sns,
        minsn=minsn,
        objtypes=objtypes,
        expid_range=expid_range,
        fit_targetid=fit_targetid,
        zbest_path=zbest_path,
        zbest_ext=zbest_ext,
        zbest_select=zbest_select,
        zbest_include=zbest_include)

    fibermap_subset_hdu = pyfits.BinTableHDU(atpy.Table(fibermap)[subset],
                                             name='FIBERMAP')
    if exp_fibermap is not None:
        tmp_sub = np.isin(exp_fibermap['TARGETID'],
                          fibermap['TARGETID'][subset])
        exp_fibermap_subset_hdu = pyfits.BinTableHDU(
            atpy.Table(exp_fibermap)[tmp_sub], name='EXP_FIBERMAP')
    scores_subset_hdu = pyfits.BinTableHDU(atpy.Table(scores)[subset],
                                           name='SCORES')

    # skip if no need to fit anything
    if not (subset.any()):
        logging.warning('No fibers selected in file %s' % (fname))
        outtab_hdus = [
            pyfits.PrimaryHDU(header=get_prim_header(
                config=config, cmdline=cmdline, zbest_path=zbest_path)),
            comment_filler(pyfits.BinTableHDU(atpy.Table([]), name='RVTAB'),
                           columnDesc), fibermap_subset_hdu, scores_subset_hdu
        ]
        if exp_fibermap is not None:
            outtab_hdus += [exp_fibermap_subset_hdu]
        outmod_hdus = [
            pyfits.PrimaryHDU(
                header=get_prim_header(config=config,
                                       cmdline=cmdline,
                                       spectrum_header=spectrum_header,
                                       zbest_path=zbest_path))
        ]

        for curs in setups:
            outmod_hdus.append(
                pyfits.ImageHDU(waves[curs],
                                name='%s_WAVELENGTH' % curs.upper()))
            outmod_hdus.append(pyfits.ImageHDU(name='%s_MODEL' % curs.upper()))

        outmod_hdus += [fibermap_subset_hdu]
        write_hdulist(mod_ofname, pyfits.HDUList(outmod_hdus))
        write_hdulist(tab_ofname, pyfits.HDUList(outtab_hdus))

        return 0
    logging.info('Selected %d fibers to fit' % (subset.sum()))

    # columns to include in the RVTAB
    columnsCopy = [
        'FIBER', 'REF_ID', 'REF_CAT', 'TARGET_RA', 'TARGET_DEC', 'TARGETID',
        'EXPID'
    ]

    outdf = []

    npixels = {}
    for curs in setups:
        npixels[curs] = fluxes[curs].shape[1]

    seqid_to_fit = np.nonzero(subset)[0]
    if rr_z is not None:
        rr_z, rr_spectype, rr_subtype = rr_z[seqid_to_fit], rr_spectype[
            seqid_to_fit], rr_subtype[seqid_to_fit]
    else:
        rr_z = np.zeros(len(seqid_to_fit)) + np.nan
        rr_spectype = np.ma.zeros(len(seqid_to_fit), dtype=str)
        rr_subtype = np.ma.zeros(len(seqid_to_fit), dtype=str)

    if use_resolution_matrix:
        sig0s = {}
        for s in setups:
            if config is not None:
                if ('lsf_sigma0_angstrom' not in config
                        or s not in config['lsf_sigma0_angstrom']):
                    cur_val = 0.5  # default value
                    logging.warning('sigma0 of the templates is not specified '
                                    f'for setup {s} using {cur_val}')
                else:
                    cur_val = config['lsf_sigma0_angstrom'][s]
                sig0s[s] = cur_val
    else:
        sig0s = None

    timers.append(time.time())
    rets = []
    nfibers_good = 0
    for (cur_rr_z, cur_rr_spectype, cur_rr_subtype,
         cur_seqid) in zip(rr_z, rr_spectype, rr_subtype, seqid_to_fit):
        # collect data
        specdatas = get_specdata(waves,
                                 fluxes,
                                 ivars,
                                 masks,
                                 resolutions,
                                 cur_seqid,
                                 setups,
                                 use_resolution_matrix=use_resolution_matrix,
                                 lsf_sigma0_angstrom=sig0s)
        cur_fibermap_row = fibermap[cur_seqid]
        if specdatas is not None:
            cur_arms = [_.name
                        for _ in specdatas]  # actual arms that are good enough
        else:
            cur_arms = None
        extra_info = {
            'fibermap_row': cur_fibermap_row,
            'seqid': cur_seqid,
            'rr_z': cur_rr_z,
            'rr_spectype': cur_rr_spectype,
            'rr_subtype': cur_rr_subtype,
            'arms': cur_arms
        }
        if specdatas is None:
            logging.warning(
                f'Giving up on fitting spectra for row {cur_fibermap_row}')
            rets.append((FakeFuture([None, None]), extra_info))
            continue
        nfibers_good += 1
        cur_brick, cur_targetid = cur_fibermap_row[
            'BRICKID'], cur_fibermap_row['TARGETID']
        if doplot:
            fig_fname = fig_prefix + '_%d_%d_%d.png' % (
                cur_brick, cur_targetid, cur_seqid)
        else:
            fig_fname = None
        rets.append((poolex.submit(
            proc_onespec, *(specdatas, setups, config, options),
            **dict(fig_fname=fig_fname, doplot=doplot,
                   ccf_init=ccf_init)), extra_info))
    timers.append(time.time())

    # This will store best-fit model data
    models = {}
    for curs in setups:
        models['desi_' + curs] = np.zeros((nfibers_good, npixels[curs]),
                                          dtype=np.float32)
    versions = None
    for ii, (r, extra_info) in enumerate(rets):
        outdict, curmodel = r.result()
        if outdict is None:
            bad_row = True
            outdict = dict(RVS_WARN=bitmasks['BAD_SPECTRUM'])
        else:
            bad_row = False
        cur_fibermap_row, cur_seqid, cur_arms = (extra_info['fibermap_row'],
                                                 extra_info['seqid'],
                                                 extra_info['arms'])
        cur_rr_z, cur_rr_spectype, cur_rr_subtype = (extra_info['rr_z'],
                                                     extra_info['rr_spectype'],
                                                     extra_info['rr_subtype'])
        for col in columnsCopy:
            if col in fibermap.columns.names:
                outdict[col] = cur_fibermap_row[col]
        for curs in setups:
            outdict['SN_%s' % curs.upper()] = sns[curs][cur_seqid]

        outdict['SUCCESS'] = outdict['RVS_WARN'] == 0
        outdict['RR_Z'] = cur_rr_z
        outdict['RR_SPECTYPE'] = cur_rr_spectype
        outdict['RR_SUBTYPE'] = cur_rr_subtype
        if not bad_row:
            for jj, curs in enumerate(cur_arms):
                models[curs][ii] = curmodel[jj]
            if 'versions' in outdict:
                if versions is None:
                    versions = outdict['versions']
                del outdict[
                    'versions']  # I don't want to store it in the table
        outdf.append(outdict)

    timers.append(time.time())
    outtab = atpy.Table(outdf)

    outmod_hdus = [
        pyfits.PrimaryHDU(
            header=get_prim_header(versions=versions,
                                   config=config,
                                   cmdline=cmdline,
                                   spectrum_header=spectrum_header,
                                   zbest_path=zbest_path))
    ]

    for curs in setups:
        outmod_hdus.append(
            pyfits.ImageHDU(waves[curs], name='%s_WAVELENGTH' % curs.upper()))
        outmod_hdus.append(
            pyfits.ImageHDU(np.vstack(models['desi_%s' % curs]),
                            name='%s_MODEL' % curs.upper()))

    outmod_hdus += [fibermap_subset_hdu]

    assert (len(fibermap_subset_hdu.data) == len(outtab))
    outtab_hdus = [
        pyfits.PrimaryHDU(header=get_prim_header(versions=versions,
                                                 config=config,
                                                 cmdline=cmdline,
                                                 zbest_path=zbest_path)),
        comment_filler(pyfits.BinTableHDU(outtab, name='RVTAB'), columnDesc),
        fibermap_subset_hdu, scores_subset_hdu
    ]
    if exp_fibermap is not None:
        outtab_hdus += [exp_fibermap_subset_hdu]
    timers.append(time.time())

    write_hdulist(mod_ofname, pyfits.HDUList(outmod_hdus))
    write_hdulist(tab_ofname, pyfits.HDUList(outtab_hdus))
    timers.append(time.time())
    logging.debug(
        str.format('Global timing: {}', (np.diff(np.array(timers)), )))
    return len(seqid_to_fit)


def write_hdulist(fname, hdulist):
    """ Write HDUList to fname
    using a temporary file which is then renamed
    """
    fname_tmp = fname + '.tmp'
    hdulist.writeto(fname_tmp, overwrite=True, checksum=True)
    os.rename(fname_tmp, fname)


def proc_desi_wrapper(*args, **kwargs):
    status = ProcessStatus.SUCCESS
    status_file = kwargs['process_status_file']
    throw_exceptions = kwargs['throw_exceptions']
    del kwargs['process_status_file']
    del kwargs['throw_exceptions']
    nfit = 0
    t1 = time.time()
    try:
        nfit = proc_desi(*args, **kwargs)
    except:  # noqa F841
        status = ProcessStatus.FAILURE
        logging.exception('failed with these arguments' + str(args) +
                          str(kwargs))
        pid = os.getpid()
        logfname = 'crash_%d_%s.log' % (pid, time.ctime().replace(' ', ''))
        with open(logfname, 'w') as fd:
            print('failed with these arguments', args, kwargs, file=fd)
            traceback.print_exc(file=fd)
        # I decided not to just fail, proceed instead after
        # writing a bug report
        if throw_exceptions:
            raise
    finally:
        t2 = time.time()
        if status_file is not None:
            if nfit < 0:
                status = ProcessStatus.FAILURE
                nfit = 0
            update_process_status_file(status_file,
                                       args[0],
                                       status,
                                       nfit,
                                       t2 - t1,
                                       start=False)


proc_desi_wrapper.__doc__ = proc_desi.__doc__


class FileQueue:
    """ This is a class that can work as an iterator
    Here we can either provide the list of files or the file with the list
    of files or use it as queue, where we pick up the top file, remove the
    line from the file and move on"""

    def __init__(self, file_list=None, file_from=None, queue=False):
        if file_list is not None:
            self.file_list = file_list
            self.file_from = None
            self.queue = False
        elif file_from is not None:
            if not queue:
                self.file_list = []
                with open(file_from, 'r') as fp:
                    for ll in fp:
                        self.file_list.append(ll.rstrip())
            else:
                self.file_list = None
                self.file_from = file_from
                self.queue = queue

    def __iter__(self):
        return self

    def __next__(self):
        if self.file_list is not None:
            if len(self.file_list) > 0:
                return self.file_list.pop(0)
            else:
                raise StopIteration
        else:
            return self.read_next()

    def read_next(self):
        import socket
        lockname = self.file_from + '.%s.%d.lock' % (socket.gethostname(),
                                                     os.getpid())
        wait_time = 1
        max_waits = 1000
        for i in range(max_waits):
            try:
                os.rename(self.file_from, lockname)
            except FileNotFoundError:
                time.sleep(np.random.uniform(wait_time, 1.5 * wait_time))
                continue
            try:
                with open(lockname, 'r') as fp1:
                    ll = fp1.readlines()
                if len(ll) == 0:
                    raise StopIteration
                ret = ll[0].rstrip()
                with open(lockname, 'w') as fp1:
                    fp1.writelines(ll[1:])
                return ret
            finally:
                os.rename(lockname, self.file_from)

        logging.warning('Cannot read next file due to lock')
        raise StopIteration


class FakeFuture:
    # this is a fake Future object designed for easier switching
    # to single thread operations when debugging
    def __init__(self, x):
        self.x = x

    def result(self):
        return self.x


class FakeExecutor:

    def __init__(self):
        pass

    def submit(self, f, *args, **kw):
        return FakeFuture(f(*args, **kw))


def get_mpi_rank():
    from mpi4py import MPI
    return MPI.COMM_WORLD.Get_rank()


def proc_many(files,
              output_dir,
              output_tab_prefix,
              output_mod_prefix,
              figure_dir=None,
              figure_prefix=None,
              config_fname=None,
              nthreads=1,
              fit_targetid=None,
              objtypes=None,
              minsn=-1e9,
              doplot=True,
              expid_range=None,
              skipexisting=False,
              fitarm=None,
              cmdline=None,
              zbest_select=False,
              zbest_include=False,
              ccf_init=True,
              subdirs=True,
              ccf_continuum_normalize=True,
              process_status_file=None,
              use_resolution_matrix=None,
              npoly=None,
              throw_exceptions=None):
    """
    Process many spectral files

    Parameters
    -----------
    files: strings
        The files with spectra
    oprefix: string
        The prefix where the table with measurements will be stored
    figure_dir: string
        The director where the figures will be stored
    figure_prefix: string
        The prefix of figure filenames
    config_fname: string
        The name of the config file
    fit_targetid: integer or None
        The targetid to fit (the rest will be ignored)
    objtypes: lists
        list of DESI_TARGET regular expressions
    doplot: bool
        Plotting
    minsn: real
        THe min S/N to fit
    cmdline: string
        The command line used in execution of rvspecfit
    expid_range: tuple
        None or a tule of two numbers for the range of expids to fit
    skipexisting: bool
        if True do not process anything if output files exist
    fitarm: list
        the list of arms/spec configurations to fit (can be None)
    npoly: integer
        the degree of the polynomial used for continuum
    process_status_file: str
        The filename where we'll put status of the fitting
    """
    override = dict(ccf_continuum_normalize=ccf_continuum_normalize)
    config = utils.read_config(config_fname, override)
    assert (config is not None)
    assert ('template_lib' in config)

    if nthreads > 1:
        parallel = True
    else:
        parallel = False
    if process_status_file is not None:
        update_process_status_file(process_status_file,
                                   None,
                                   None,
                                   None,
                                   None,
                                   start=True)
    if parallel:
        poolEx = concurrent.futures.ProcessPoolExecutor(nthreads)
    else:
        poolEx = FakeExecutor()
    for f in files:
        fname = f.split('/')[-1]
        if subdirs:
            assert (len(f.split('/')) > 2)
            # we need that because we use the last two directories in the path
            # to create output directory structure
            # i.e. input file a/b/c/d/e/f/g.fits will produce output file in
            # output_prefix/e/f/xxx.fits
            fdirs = f.split('/')
            folder_path = output_dir + '/' + fdirs[-3] + '/' + fdirs[-2] + '/'
        else:
            folder_path = output_dir + '/'
        os.makedirs(folder_path, exist_ok=True)
        logging.debug(f'Making folder {folder_path}')
        if figure_dir is not None:
            if subdirs:
                figure_path = figure_dir + '/' + fdirs[-3] + '/' + fdirs[
                    -2] + '/'
            else:
                figure_path = figure_dir + '/'
            os.makedirs(figure_path, exist_ok=True)
            cur_figure_prefix = figure_path + '/' + figure_prefix
            logging.debug(f'Making folder {figure_path}')
        else:
            cur_figure_prefix = None
        if fname[-3:] == '.gz':
            fname0 = fname[:-3]
        else:
            fname0 = fname
        tab_ofname = folder_path + output_tab_prefix + '_' + fname0
        mod_ofname = folder_path + output_mod_prefix + '_' + fname0

        if (skipexisting and os.path.exists(tab_ofname)
                and os.path.exists(mod_ofname)):
            logging.info('skipping, products already exist %s' % f)
            if process_status_file is not None:
                update_process_status_file(process_status_file, f,
                                           ProcessStatus.EXISTING, -1, 0)

            continue
        args = (f, tab_ofname, mod_ofname, cur_figure_prefix, config)
        kwargs = dict(fit_targetid=fit_targetid,
                      objtypes=objtypes,
                      doplot=doplot,
                      minsn=minsn,
                      expid_range=expid_range,
                      poolex=poolEx,
                      fitarm=fitarm,
                      cmdline=cmdline,
                      zbest_select=zbest_select,
                      zbest_include=zbest_include,
                      process_status_file=process_status_file,
                      npoly=npoly,
                      ccf_init=ccf_init,
                      use_resolution_matrix=use_resolution_matrix,
                      throw_exceptions=throw_exceptions)
        proc_desi_wrapper(*args, **kwargs)

    if parallel:
        try:
            poolEx.shutdown(wait=True)
        except KeyboardInterrupt:
            poolEx.shutdown(wait=False)
            raise

    logging.info("Successfully finished processing")


def main(args):
    cmdline = ' '.join(args)
    parser = argparse.ArgumentParser()

    parser.add_argument('--nthreads',
                        help='Number of threads for the fits',
                        type=int,
                        default=1)

    parser.add_argument('--config',
                        help='The filename of the configuration file',
                        type=str,
                        default=None)

    parser.add_argument('input_files',
                        help='Space separated list of files to process',
                        type=str,
                        default=None,
                        nargs='*')
    parser.add_argument(
        '--input_file_from',
        help='Read the list of spectral files from the text file',
        type=str,
        default=None)
    parser.add_argument(
        '--queue_file',
        help='If the input file list is a queue where we delete entries'
        'as soon as we picked up a file',
        action='store_true',
        default=False)

    parser.add_argument('--output_dir',
                        help='Output directory for the tables',
                        type=str,
                        default='./',
                        required=False)
    parser.add_argument('--targetid',
                        help='Fit only a given targetid',
                        type=int,
                        default=None,
                        required=False)
    parser.add_argument('--targetid_file_from',
                        help='Fit only a given targetids from a given file',
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument('--minsn',
                        help='Fit only S/N larger than this',
                        type=float,
                        default=-1e9,
                        required=False)
    parser.add_argument('--minexpid',
                        help='Min expid',
                        type=int,
                        default=None,
                        required=False)
    parser.add_argument('--maxexpid',
                        help='Max expid',
                        type=int,
                        default=None,
                        required=False)

    parser.add_argument('--npoly',
                        help='npoly',
                        type=int,
                        default=None,
                        required=False)

    parser.add_argument('--output_tab_prefix',
                        help='Prefix of output table files',
                        type=str,
                        default=GlobalConfig.table_prefix,
                        required=False)
    parser.add_argument('--output_mod_prefix',
                        help='Prefix of output model files',
                        type=str,
                        default=GlobalConfig.model_prefix,
                        required=False)
    parser.add_argument('--fitarm',
                        help='Comma separated arms like b,r,z or b ',
                        type=str,
                        required=False)
    parser.add_argument('--figure_dir',
                        help='Path for the fit figures, i.e. fig_folder/',
                        type=str,
                        default='./')
    parser.add_argument('--figure_prefix',
                        help='Prefix for the fit figures filename, i.e. im',
                        type=str,
                        default='fig',
                        required=False)
    parser.add_argument('--log',
                        help='Log filename',
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument('--log_level',
                        help='DEBUG/INFO/WARNING/ERROR/CRITICAL',
                        type=str,
                        default='WARNING',
                        required=False)
    parser.add_argument('--param_init',
                        help='How the initial parameters/RV are initialized',
                        type=str,
                        default='CCF',
                        required=False)
    parser.add_argument('--process_status_file',
                        help='The name of the file where I put the names of' +
                        ' successfully processed files',
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument('--resolution_matrix', action='store_true')
    parser.add_argument('--no-resolution_matrix',
                        dest='resolution_matrix',
                        action='store_false')
    parser.set_defaults(resolution_matrix=False)

    parser.add_argument(
        '--overwrite',
        help='''If enabled the code will overwrite the existing products,
 otherwise it will attempt to update/append''',
        default=None,
        required=False)
    parser.add_argument('--version',
                        help='Output the version of the software',
                        action='store_true',
                        default=False)

    parser.add_argument('--skipexisting',
                        help='''If enabled the code will completely skip
 if there are existing products''',
                        action='store_true',
                        default=False)

    parser.add_argument(
        '--zbest_select',
        help='''If enabled the code will try to use the zbest file to fit \
only potentially interesting targets''',
        action='store_true',
        default=False)
    parser.add_argument(
        '--zbest_include',
        help='''If enabled the code will include the zbest/redrock info \
in the table (but will not use for selection)''',
        action='store_true',
        default=False)

    parser.add_argument('--doplot',
                        help='Make plots',
                        action='store_true',
                        default=False)

    parser.add_argument('--mpi',
                        help='Use MPI for communication',
                        action='store_true',
                        default=False)

    parser.add_argument(
        '--no_ccf_continuum_normalize',
        help='Do not normalize by the continuum when doing CCF',
        dest='ccf_continuum_normalize',
        action='store_false',
        default=True)
    parser.add_argument(
        '--no_subdirs',
        help='Do not create the subdirectories in the output dir',
        dest='subdirs',
        action='store_false',
        default=True)

    parser.add_argument('--throw_exceptions',
                        help='If this option is set, the code will not'
                        ' protect against exceptions inside rvspecfit',
                        action='store_true',
                        default=False)

    parser.add_argument('--objtypes',
                        help='The list of targets MWS_ANY,SCND_ANY,STD_*',
                        type=str,
                        default=None)
    parser.set_defaults(ccf_continuum_normalize=True)
    args = parser.parse_args(args)

    if args.version:
        print(rvspecfit._version.version)
        sys.exit(0)

    log_level = args.log_level

    log_filename = args.log
    process_status_file = args.process_status_file
    if log_filename is not None:
        if args.mpi:
            if re.match('.*%.*', log_filename) is None:
                raise RuntimeError(
                    'If using MPI and log files you need to '
                    'allow for the MPI rank to placed in the filename using %d'
                )
            log_filename = log_filename % (get_mpi_rank())
        logging.basicConfig(filename=log_filename, level=log_level)
    else:
        logging.basicConfig(level=log_level)

    if process_status_file is not None:
        if args.mpi:
            if re.match('.*%.*', process_status_file) is None:
                raise RuntimeError(
                    'If using MPI and log files you need to '
                    'allow for the MPI rank to placed in the filename using %d'
                )
            process_status_file = process_status_file % (get_mpi_rank())

    input_files = args.input_files
    input_file_from = args.input_file_from
    output_dir, output_tab_prefix, output_mod_prefix = (args.output_dir,
                                                        args.output_tab_prefix,
                                                        args.output_mod_prefix)
    queue_file = args.queue_file
    nthreads = args.nthreads
    config_fname = args.config
    doplot = args.doplot
    zbest_select = args.zbest_select
    zbest_include = args.zbest_include
    minsn = args.minsn
    objtypes = args.objtypes
    if objtypes is not None:
        objtypes = objtypes.split(',')
    minexpid = args.minexpid
    maxexpid = args.maxexpid
    targetid_file_from = args.targetid_file_from
    targetid = args.targetid
    npoly = args.npoly
    fitarm = args.fitarm
    if fitarm is not None:
        fitarm = fitarm.split(',')
        fitarm = [_.lower() for _ in fitarm]
        for _ in fitarm:
            if _ not in 'brz':
                raise ValueError('only allowed arm names are brz')
    if args.param_init == 'CCF':
        ccf_init = True
    elif args.param_init == 'bruteforce':
        ccf_init = False
    else:
        raise ValueError(
            'Unknown param_init value; only known ones are CCF and bruteforce')
    ccf_continuum_normalize = args.ccf_continuum_normalize

    fit_targetid = None
    if targetid_file_from is not None and targetid is not None:
        raise RuntimeError(
            'You can only specify targetid or targetid_file_from options')
    elif targetid_file_from is not None:
        fit_targetid = []
        with open(targetid_file_from, 'r') as fp:
            for curl in fp:
                fit_targetid.append(int(curl.rstrip()))
        fit_targetid = np.unique(fit_targetid)
    elif targetid is not None:
        fit_targetid = np.unique([targetid])
    else:
        pass
    if doplot:
        figure_dir = args.figure_dir
    else:
        figure_dir = None

    if args.overwrite is not None:
        logging.warning('overwrite keyword is meaningless now')

    # Dealing with input
    if args.mpi:
        rank = get_mpi_rank()
    else:
        rank = 0
    if (not args.mpi) or (args.mpi and rank == 0):
        # if root process in mpi mode or
        # or anything non-mpi
        if input_files != [] and input_file_from is not None:
            raise RuntimeError('You can only specify --input_files OR '
                               '--input_file_from options '
                               'but not both of them simultaneously')
        elif input_file_from is None and input_files == []:
            parser.print_help()
            raise RuntimeError(
                'You need to specify the spectra you want to fit')
        if input_files == []:
            input_files = None
    else:
        # only in the case of mpi and rank>=1
        input_files = None
    if not args.mpi:
        files = FileQueue(file_list=input_files,
                          file_from=input_file_from,
                          queue=queue_file)
    else:
        if input_files is None:
            with open(input_file_from, 'r') as fp:
                input_files = [_.rstrip() for _ in fp.readlines()]
        files = utils.MPIFileQueue(file_list=input_files)

    if (not args.mpi) or (args.mpi and rank != 0):
        # anything but mpi and rank=0 should go here
        proc_many(
            files,
            output_dir,
            output_tab_prefix,
            output_mod_prefix,
            figure_dir=figure_dir,
            figure_prefix=args.figure_prefix,
            nthreads=nthreads,
            config_fname=config_fname,
            fit_targetid=fit_targetid,
            objtypes=objtypes,
            doplot=doplot,
            subdirs=args.subdirs,
            minsn=minsn,
            process_status_file=process_status_file,
            expid_range=(minexpid, maxexpid),
            skipexisting=args.skipexisting,
            fitarm=fitarm,
            cmdline=cmdline,
            zbest_select=zbest_select,
            zbest_include=zbest_include,
            ccf_continuum_normalize=ccf_continuum_normalize,
            use_resolution_matrix=args.resolution_matrix,
            ccf_init=ccf_init,
            npoly=npoly,
            throw_exceptions=args.throw_exceptions,
        )
    else:
        files.distribute_files()


if __name__ == '__main__':
    main(sys.argv[1:])

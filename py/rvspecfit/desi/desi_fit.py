import os

os.environ['OMP_NUM_THREADS'] = '1'

# The noqa are to prevent warnings due to imports being not at the top
import warnings  # noqa: E402
import sys  # noqa: E402
import argparse  # noqa: E402
import time  # noqa: E402
import itertools  # noqa: E402
import traceback  # noqa: E402
import importlib  # noqa: E402
import logging  # noqa: E402
import enum  # noqa: E402
import concurrent.futures  # noqa: E402
import astropy.table as atpy  # noqa: E402
import astropy.io.fits as pyfits  # noqa: E402
import astropy.units as auni  # noqa: E402
import numpy as np  # noqa: E402
import scipy.stats  # noqa: E402
import rvspecfit  # noqa: E402
from rvspecfit import fitter_ccf, vel_fit, spec_fit, utils, \
    spec_inter  # noqa: E402


class ProcessStatus(enum.Enum):
    SUCCESS = 0
    FAILURE = 1
    EXISTING = 2

    def __str__(self):
        return self.name


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
    packages = [
        'numpy', 'astropy', 'matplotlib', 'rvspecfit', 'pandas', 'scipy',
        'yaml', 'numdifftools'
    ]
    # Ideally you need to check that the list here matches the requirements.txt
    ret = {}
    for curp in packages:
        ret[curp] = importlib.import_module(curp).__version__
    ret['python'] = str.split(sys.version, ' ')[0]
    return ret


def get_zbest_fname(fname):
    """ Get the zbest file for a given spectrum """
    paths = fname.split('/')
    fname_end = paths[-1]

    f1 = fname_end.replace('coadd-', 'zbest-')
    if f1 == fname_end:
        f1 = fname_end.replace('spectra-', 'zbest-')
        if f1 == fname_end:
            return None
    zbest_path = '/'.join(paths[:-1] + [f1])
    if os.path.exists(zbest_path):
        return zbest_path
    return zbest_path


def get_prim_header(versions={}, config=None, cmdline=None):
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


def valid_file(fname):
    """
    Check if all required extensions are present if yes return true
    """
    exts = pyfits.open(fname)
    extnames = [_.name for _ in exts]

    names0 = []
    arms = ['B', 'R', 'Z']
    arm_glued = 'BRZ'
    prefs = 'WAVELENGTH', 'FLUX', 'IVAR', 'MASK'
    reqnames = names0 + [
        '%s_%s' % (_, __) for _, __ in itertools.product(arms, prefs)
    ]
    reqnames_glued = names0 + ['%s_%s' % (arm_glued, _) for _ in prefs]
    missing = []
    for curn in reqnames:
        if curn not in extnames:
            missing.append(curn)
    if len(missing) != 0:
        missing_glued = []
        for curn in reqnames_glued:
            if curn not in extnames:
                missing_glued.append(curn)
        if len(missing_glued) > 0:
            logging.warning('Extensions %s are missing' % (','.join(missing)))
            return (False, True)
        return (True, True)
    return (True, False)


def proc_onespec(
    specdata,
    setups,
    config,
    options,
    fig_fname='fig.png',
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

    Returns
    -------
    outdict: dict
        Dictionary with fit measurements
    yfit: list
        List of best-fit models
    """
    chisqs = {}
    chisqs_c = {}
    t1 = time.time()
    res = fitter_ccf.fit(specdata, config)
    t2 = time.time()
    paramDict0 = res['best_par']
    fixParam = []
    if res['best_vsini'] is not None:
        paramDict0['vsini'] = min(max(res['best_vsini'], config['min_vsini']),
                                  config['max_vsini'])

    res1 = vel_fit.process(
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
    outdict = dict(
        VRAD=res1['vel'] * auni.km / auni.s,
        VRAD_ERR=res1['vel_err'] * auni.km / auni.s,
        VRAD_SKEW=res1['vel_skewness'],
        VRAD_KURT=res1['vel_kurtosis'],
        LOGG=res1['param']['logg'],
        TEFF=res1['param']['teff'] * auni.K,
        ALPHAFE=res1['param']['alpha'],
        FEH=res1['param']['feh'],
        LOGG_ERR=res1['param_err']['logg'],
        TEFF_ERR=res1['param_err']['teff'] * auni.K,
        ALPHAFE_ERR=res1['param_err']['alpha'],
        FEH_ERR=res1['param_err']['feh'],
        VSINI=res1['vsini'] * auni.km / auni.s,
        NEXP=len(specdata) // len(setups),
    )

    for i, curd in enumerate(specdata):
        if curd.name not in chisqs:
            chisqs[curd.name] = 0
            chisqs_c[curd.name] = 0
        chisqs[curd.name] += res1['chisq_array'][i]
        chisqs_c[curd.name] += chisq_cont_array[i]

    outdict['CHISQ_TOT'] = sum(chisqs.values())
    outdict['CHISQ_C_TOT'] = sum(chisqs_c.values())

    for s in chisqs.keys():
        outdict['CHISQ_%s' % s.replace('desi_', '').upper()] = chisqs[s]
        outdict['CHISQ_C_%s' % s.replace('desi_', '').upper()] = float(
            chisqs_c[s])

    chisq_thresh = 50
    # if the delta-chisq between continuum is smaller than this we
    # set a warning flag

    rvedge_thresh = 5 * auni.km / auni.s
    # if we are within this threshold of the RV boundary we set another
    # warning
    rverr_thresh = 100 * auni.km / auni.s
    # If the error is larger than this we warn

    rvs_warn = 0
    bitmasks = {'CHISQ_WARN': 1, 'RV_WARN': 2, 'RVERR_WARN': 4}
    dchisq = outdict['CHISQ_C_TOT'] - outdict['CHISQ_TOT']  # should be >0

    if (dchisq < chisq_thresh):
        rvs_warn |= bitmasks['CHISQ_WARN']

    if (np.abs(outdict['VRAD'] - config['max_vel'] * auni.km / auni.s) <
            rvedge_thresh) or (np.abs(outdict['VRAD'] - config['min_vel'] *
                                      auni.km / auni.s) < rvedge_thresh):
        rvs_warn |= bitmasks['RV_WARN']
    if (outdict['VRAD_ERR'] > rverr_thresh):
        rvs_warn |= bitmasks['RVERR_WARN']

    outdict['RVS_WARN'] = rvs_warn

    if doplot:
        title = ('logg=%.1f teff=%.1f [Fe/H]=%.1f ' +
                 '[alpha/Fe]=%.1f Vrad=%.1f+/-%.1f') % (
                     res1['param']['logg'], res1['param']['teff'],
                     res1['param']['feh'], res1['param']['alpha'], res1['vel'],
                     res1['vel_err'])
        if len(specdata) > len(setups):
            for i in range(len(specdata) // len(setups)):
                sl = slice(i * len(setups), (i + 1) * len(setups))
                make_plot(specdata[sl], res1['yfit'][sl], title,
                          fig_fname.replace('.png', '_%d.png' % i))
        else:
            make_plot(specdata, res1['yfit'], title, fig_fname)
    versions = {}
    for i, (k, v) in enumerate(spec_inter.interp_cache.interps.items()):
        versions[k] = dict(revision=v.revision,
                           creation_soft_version=v.creation_soft_version)
    outdict['versions'] = versions
    logging.debug('Timing: %.4f %.4f %.4f' % (t2 - t1, t3 - t2, t4 - t3))
    return outdict, res1['yfit']


def get_sns(data, ivars, masks):
    """
    Return the vector of S/Ns
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xind = (ivars <= 0) | (masks > 0)
        xsn = data * np.sqrt(ivars)
        xsn[xind] = np.nan
        xsn[xind] = np.nan
        sns = np.nanmedian(xsn, axis=1)
        sns[~np.isfinite(sns)] = -1e9  # set problematic spectrum sn to
        # a very negative number
    return sns


def read_data(fname, glued, setups):
    """ Read the data file

    Parameters
    ----------
    fname: str
        Filename
    glued: bool
        True if BRZ format
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
    for s in setups:
        fluxes[s] = pyfits.getdata(fname, '%s_FLUX' % s.upper())
        ivars[s] = pyfits.getdata(fname, '%s_IVAR' % s.upper())
        masks[s] = pyfits.getdata(fname, '%s_MASK' % s.upper())
        waves[s] = pyfits.getdata(fname, '%s_WAVELENGTH' % s.upper())
    return fluxes, ivars, masks, waves


def select_fibers_to_fit(fibermap,
                         sns,
                         zbest_path=None,
                         minsn=None,
                         mwonly=True,
                         expid_range=None,
                         glued=False,
                         fit_targetid=None,
                         zbest_select=False):
    """ Identify fibers to fit
    Currently that either uses MWS_TARGET or S/N cut

    Parameters
    ----------
    fibermap: Table
        Fibermap table object
    sns: dict of numpy arrays
        Array of S/Ns
    minsn: float
        Threshold S/N
    mwonly: bool
        Only fit MWS
    expid_range: list
        The range of EXPID to consider
    glued: bool
        If the data has been BRZ glued (deprecated)
    fit_targetid: list of ints
        Fit only specific TARGETIDs

    Returns
    -------
    ret: bool numpy array
        Array with True for selected spectra

    """
    if mwonly:
        subset = fibermap['MWS_TARGET'] != 0
    else:
        subset = np.ones(len(fibermap), dtype=bool)
    if expid_range is not None:
        mine, maxe = expid_range
        if mine is None:
            mine = -1
        if maxe is None:
            maxe = np.inf
    if not glued:
        if "EXPID" in fibermap.columns.names:
            subset = subset & (fibermap["EXPID"] > mine) & (fibermap['EXPID']
                                                            <= maxe)
    if fit_targetid is not None:
        subset = subset & np.in1d(fibermap['TARGETID'], fit_targetid)
    if minsn is not None:
        maxsn = np.max(np.array(list(sns.values())), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            subset = subset & (maxsn > minsn)
    if zbest_select:
        if zbest_path is None:
            warnings.warn(
                'ZBest selection requested, but the zbest file not found')
        else:
            zb = atpy.Table().read(zbest_path, format='fits', hdu='ZBEST')
            assert (len(zb) == len(subset))
            subset = (((zb['SPECTYPE'] == 'STAR') |
                       (np.abs(zb['Z']) < 2000 / 3e5))) & subset
    return subset


def get_unique_seqid_to_fit(targetid, subset, combine=False):
    """
    Return the row ids of that needs to be processed
    The complexity here is dealing with the combine mode, in that
    case I return list of lists of integers

    """
    if not combine:
        return np.nonzero(subset)[0]

    seqid = np.arange(len(targetid))
    utargetid, inv = np.unique(targetid)
    ret = []
    for u in utargetid:
        ret.append(seqid[targetid == u])
    return ret


def get_specdata(waves, fluxes, ivars, masks, seqid, glued, setups):
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
    glued: bool
        If spectrum is in BRZ format
    setups: list
        List of configurations

    Returns
    -------
    ret: list
        List of specfit.SpecData objects or None if failed

    """
    large_error = 1000
    sds = []
    minerr_frac = 0.3  # if the error is smaller than this times median error
    # clamp the uncertainty
    for s in setups:
        spec = fluxes[s][seqid] * 1
        curivars = ivars[s][seqid] * 1
        medspec = np.nanmedian(spec)
        if medspec == 0:
            medspec = np.nanmedian(spec[spec > 0])
            if not np.isfinite(medspec):
                medspec = np.nanmedian(np.abs(spec))
        if not np.isfinite(medspec) or medspec == 0:
            # bail out the spectrum is insane
            # TODO make the logic clearer
            return None
        baddat = ~np.isfinite(spec + curivars)
        badmask = (masks[s][seqid] > 0)
        baderr = curivars <= 0
        badall = baddat | badmask | baderr
        curivars[badall] = 1. / medspec**2 / large_error**2
        spec[badall] = medspec
        espec = 1. / curivars**.5
        if badall.all():
            logging.warning('The whole spectrum was masked...')
        else:
            goodespec = espec[~badall]
            goodespec_thresh = np.median(goodespec) * minerr_frac
            replace_idx = (espec < goodespec_thresh) & (~badall)
            if replace_idx.sum() / (~badall).sum() > .1:
                logging.warning(
                    'More than 10% of spectra had the uncertainty clamped')
            # logging.debug("Clamped error on %d pixels" % (replace_idx.sum()))
            espec[replace_idx] = goodespec_thresh

        sd = spec_fit.SpecData('desi_%s' % s,
                               waves[s],
                               spec,
                               espec,
                               badmask=badall)
        sds.append(sd)
    return sds


def comment_filler(tab, desc):
    """ Fill comments in the FITS header """
    for i, k in enumerate(tab.data.columns):
        tab.header['TCOMM%d' % (i + 1)] = desc.get(k.name) or ''
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
        ('VRAD', 'Radial velocity'), ('VRAD_ERR', 'Radial velocity error'),
        ('VRAD_SKEW', 'Radial velocity posterior skewness'),
        ('VRAD_KURT', 'Radial velocity posterior kurtosis'),
        ('VSINI', 'Stellar rotation velocity'),
        ('LOGG', 'Log of surface gravity'), ('TEFF', 'Effective temperature'),
        ('FEH', '[Fe/H] from template fitting'),
        ('ALPHAFE', '[alpha/Fe] from template fitting'),
        ('LOGG_ERR', 'Log of surface gravity uncertainty'),
        ('TEFF_ERR', 'Effective temperature uncertainty'),
        ('FEH_ERR', '[Fe/H] uncertainty from template fitting'),
        ('ALPHAFE_ERR', '[alpha/Fe] uncertainty from template fitting'),
        ('CHISQ_TOT', 'Total chi-square for all arms'),
        ('CHISQ_C_TOT',
         'Total chi-square for all arms for polynomial only fit'),
        ('TARGETID', 'DESI targetid'), ('EXPID', 'DESI exposure id'),
        ('SUCCESS', "Did we succeed or fail"),
        ('RVS_WARN', "RVSpecFit warning flag")
    ])

    for curs in setups:
        curs = curs.upper()
        columnDesc['SN_%s' % curs] = ('Median S/N in the %s arm' % curs)
        columnDesc['CHISQ_%s' % curs] = ('Chi-square in the %s arm' % curs)
        columnDesc['CHISQ_C_%s' % curs] = (
            'Chi-square in the %s arm after fitting continuum only' % curs)
    return columnDesc


def proc_desi(fname,
              tab_ofname,
              mod_ofname,
              fig_prefix,
              config,
              fit_targetid=None,
              combine=False,
              mwonly=True,
              doplot=True,
              minsn=-1e9,
              expid_range=None,
              overwrite=False,
              poolex=None,
              fitarm=None,
              cmdline=None,
              zbest_select=False):
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
    mwonly: bool
        Fit only MWS_TARGET or every object
    doplot: bool
        Produce plots
    minsn: real
        The slallest S/N for processing
    expid_range: tuple of ints
        The range of expids to fit
    overwrite: bool
        If true, the output file will be always be overwritten, otherwise
        the results may be appended instead
    fitarm: list
         List of strings corresponding to configurations that need to be fit,
         it can be none
    zbest_select: bool
         If true then the zbest file is used to preselect targets
    poolex: Executor
         The executor that will run parallel processes
    """

    options = {'npoly': 10}
    timers = []
    timers.append(time.time())
    logging.info('Processing %s' % fname)
    valid, glued = valid_file(fname)
    if not valid:
        logging.error('Not valid file: %s' % (fname))
        return -1

    if glued:
        setups = ['brz']
    else:
        setups = ['b', 'r', 'z']
    if fitarm is not None:
        setups = [_ for _ in setups if _ in fitarm]
        assert (len(setups) > 0)

    fibermap = pyfits.getdata(fname, 'FIBERMAP')
    fluxes, ivars, masks, waves = read_data(fname, glued, setups)
    sns = dict([(_, get_sns(fluxes[_], ivars[_], masks[_])) for _ in setups])

    for _ in setups:
        if len(sns[_]) != len(fibermap):
            logging.warning((
                'WARNING the size of the data in arm %s' +
                'does not match the size of the fibermap; file %s; skipping...'
            ) % (_, fname))
            return -1
    targetid = fibermap['TARGETID']
    zbest_path = get_zbest_fname(fname)
    subset = select_fibers_to_fit(fibermap,
                                  sns,
                                  minsn=minsn,
                                  mwonly=mwonly,
                                  expid_range=expid_range,
                                  glued=glued,
                                  fit_targetid=fit_targetid,
                                  zbest_path=zbest_path,
                                  zbest_select=zbest_select)

    # skip if no need to fit anything
    if not (subset.any()):
        logging.warning('No fibers selected in file %s' % (fname))
        put_empty_file(tab_ofname)
        put_empty_file(mod_ofname)
        return 0
    logging.debug('Selected %d fibers to fit' % (subset.sum()))
    # if we are combining

    # columns to include in the RVTAB
    columnsCopy = ['FIBER', 'REF_ID', 'TARGET_RA', 'TARGET_DEC', 'TARGETID']
    if not glued:
        columnsCopy.append('EXPID')

    outdf = []

    # This will store best-fit model data
    models = {}
    for curs in setups:
        models['desi_%s' % curs] = []

    seqid_to_fit = get_unique_seqid_to_fit(targetid, subset, combine=combine)
    subset_ret = subset.copy()
    # returned subset to deal with the fact that
    # I'll skip some spectra
    # in the future I should instead fill things with nans
    # TODO

    timers.append(time.time())
    rets = []
    nfibers_good = 0
    for curseqid in seqid_to_fit:
        # collect data (if combining that means multiple spectra)
        if not combine:
            specdatas = get_specdata(waves, fluxes, ivars, masks, curseqid,
                                     glued, setups)
            curFiberRow = fibermap[curseqid]
        else:
            specdatas = sum([
                get_specdata(waves, fluxes, ivars, masks, _, glued, setups)
                for _ in curseqid
            ], [])
            curFiberRow = fibermap[curseqid[0]]
        if specdatas is None:
            logging.warning(
                f'Giving up on fitting spectra for row {curFiberRow}')
            subset_ret[curseqid] = False
            continue
        nfibers_good += 1
        curbrick, curtargetid = curFiberRow['BRICKID'], curFiberRow['TARGETID']
        fig_fname = fig_prefix + '_%d_%d_%d.png' % (curbrick, curtargetid,
                                                    curseqid)

        rets.append(
            (poolex.submit(proc_onespec, *(specdatas, setups, config, options),
                           **dict(fig_fname=fig_fname,
                                  doplot=doplot)), curFiberRow, curseqid))
    timers.append(time.time())
    if nfibers_good == 0:
        logging.warning('In the end no spectra worth fitting...')
        put_empty_file(tab_ofname)
        put_empty_file(mod_ofname)
        return 0
    for r in rets:
        outdict, curmodel = r[0].result()
        versions = outdict['versions']
        del outdict['versions']  # I don't want to store it in the table
        curFiberRow, curseqid = r[1], r[2]

        for col in columnsCopy:
            if col in fibermap.columns.names:
                outdict[col] = curFiberRow[col]
        for curs in setups:
            outdict['SN_%s' % curs.upper()] = sns[curs][curseqid]

        outdict['SUCCESS'] = True

        outdf.append(outdict)

        for ii, curs in enumerate(setups):
            # I assume all the setusp were fitted
            models['desi_%s' % curs].append(curmodel[ii])
    timers.append(time.time())
    outdf1 = {}
    for k in outdf[0].keys():
        # we can't just concatenate quantities easily
        if isinstance(outdf[0][k], auni.Quantity):
            outdf1[k] = auni.Quantity([_[k] for _ in outdf])
        else:
            outdf1[k] = np.array([_[k] for _ in outdf])
    outtab = atpy.Table(outdf1)
    fibermap_subset_hdu = pyfits.BinTableHDU(atpy.Table(fibermap)[subset_ret],
                                             name='FIBERMAP')
    outmod_hdus = [
        pyfits.PrimaryHDU(header=get_prim_header(
            versions=versions, config=config, cmdline=cmdline))
    ]

    # TODO
    # in the combine mode I don't know how to write the model
    #

    for curs in setups:
        outmod_hdus.append(
            pyfits.ImageHDU(waves[curs], name='%s_WAVELENGTH' % curs.upper()))
        outmod_hdus.append(
            pyfits.ImageHDU(np.vstack(models['desi_%s' % curs]),
                            name='%s_MODEL' % curs.upper()))

    columnDesc = get_column_desc(setups)

    outmod_hdus += [fibermap_subset_hdu]
    assert (len(fibermap_subset_hdu.data) == len(outtab))
    outtab_hdus = [
        pyfits.PrimaryHDU(header=get_prim_header(
            versions=versions, config=config, cmdline=cmdline)),
        comment_filler(pyfits.BinTableHDU(outtab, name='RVTAB'), columnDesc),
        fibermap_subset_hdu
    ]
    timers.append(time.time())

    old_rvtab = None
    if os.path.exists(tab_ofname):
        try:
            old_rvtab = atpy.Table().read(tab_ofname,
                                          format='fits',
                                          hdu='FIBERMAP')
        except (FileNotFoundError, OSError, ValueError) as e:  # noqa F841
            pass

    if overwrite or old_rvtab is None:
        # if output files do not exist or I cant read fibertab
        pyfits.HDUList(outmod_hdus).writeto(mod_ofname,
                                            overwrite=True,
                                            checksum=True)
        pyfits.HDUList(outtab_hdus).writeto(tab_ofname,
                                            overwrite=True,
                                            checksum=True)

    else:
        refit_tab = atpy.Table(fibermap)[subset_ret]
        # find a replacement subset
        refit_tab_pd = refit_tab.to_pandas()
        old_rvtab_pd = old_rvtab.to_pandas()
        refit_tab_pd['rowid'] = np.arange(len(refit_tab), dtype=int)
        old_rvtab_pd['rowid'] = np.arange(len(old_rvtab), dtype=int)

        if glued:
            pkey = ['TARGETID']
        else:
            pkey = ['EXPID', 'TARGETID']
        merge = refit_tab_pd.merge(old_rvtab_pd,
                                   left_on=pkey,
                                   right_on=pkey,
                                   suffixes=['_x', '_y'],
                                   indicator=True,
                                   how='outer')
        repset = np.array(merge['_merge'] == 'both_only')
        repid_old = np.array(merge['rowid_y'], dtype=int)[repset]

        keepmask = np.ones(len(old_rvtab), dtype=bool)
        keepmask[repid_old] = False
        # this is the subset of the old data that must
        # be kept
        merge_hdus(outmod_hdus, mod_ofname, keepmask, columnDesc, glued,
                   setups)
        merge_hdus(outtab_hdus, tab_ofname, keepmask, columnDesc, glued,
                   setups)
    timers.append(time.time())
    logging.debug(
        str.format('Global timing: {}', (np.diff(np.array(timers)), )))
    return len(seqid_to_fit)


def merge_hdus(hdus, ofile, keepmask, columnDesc, glued, setups):
    allowed = ['FIBERMAP', 'RVTAB'] + [
        '%s_MODEL' % _.upper() for _ in setups
    ] + ['%s_WAVELENGTH' % _.upper() for _ in setups]

    for i in range(len(hdus)):
        if i == 0:
            continue
        curhdu = hdus[i]
        curhduname = curhdu.name

        if curhduname not in allowed:
            raise Exception('Weird extension', curhduname)
        if curhduname in ['FIBERMAP', 'RVTAB']:
            newdat = atpy.Table(curhdu.data)
            olddat = atpy.Table().read(ofile, hdu=curhduname)
            tab = atpy.vstack((olddat[keepmask], newdat))
            curouthdu = comment_filler(
                pyfits.BinTableHDU(tab, name=curhduname), columnDesc)
            hdus[i] = curouthdu
            continue
        if curhduname[-10:] == 'WAVELENGTH':
            continue
        if curhduname[-5:] == 'MODEL':
            newdat = curhdu.data
            olddat = pyfits.getdata(ofile, curhduname)
            hdus[i] = pyfits.ImageHDU(np.concatenate(
                (olddat[keepmask], newdat), axis=0),
                                      name=curhduname)
            continue
        raise Exception('I should not be here')

    ofile_tmp = ofile + '.tmp'
    pyfits.HDUList(hdus).writeto(ofile_tmp, overwrite=True, checksum=True)
    os.rename(ofile_tmp, ofile)


def proc_desi_wrapper(*args, **kwargs):
    status = ProcessStatus.SUCCESS
    status_file = kwargs['process_status_file']
    del kwargs['process_status_file']
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


def proc_many(
    files,
    output_dir,
    output_tab_prefix,
    output_mod_prefix,
    fig_prefix,
    config_fname=None,
    nthreads=1,
    combine=False,
    fit_targetid=None,
    mwonly=True,
    minsn=-1e9,
    doplot=True,
    expid_range=None,
    overwrite=False,
    skipexisting=False,
    fitarm=None,
    cmdline=None,
    zbest_select=False,
    process_status_file=None,
):
    """
    Process many spectral files

    Parameters
    -----------
    files: strings
        The files with spectra
    oprefix: string
        The prefix where the table with measurements will be stored
    fig_prefix: string
        The prfix where the figures will be stored
    config_fname: string
        The name of the config file
    combine: bool
        Fit spectra of same targetid together
    fit_targetid: integer or None
        The targetid to fit (the rest will be ignored)
    mwonly: bool
        Only fit mws_target
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
    process_status_file: str
        The filename where we'll put status of the fitting
    """
    config = utils.read_config(config_fname)
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
    res = []
    for f in files:
        fname = f.split('/')[-1]
        assert (len(f.split('/')) > 2)
        # we need that because we use the last two directories in the path
        # to create output directory structure
        # i.e. input file a/b/c/d/e/f/g.fits will produce output file in
        # output_prefix/e/f/xxx.fits
        fdirs = f.split('/')
        folder_path = output_dir + '/' + fdirs[-3] + '/' + fdirs[-2] + '/'
        os.makedirs(folder_path, exist_ok=True)
        logging.debug(f'Making folder {folder_path}')
        tab_ofname = folder_path + output_tab_prefix + '_' + fname
        mod_ofname = folder_path + output_mod_prefix + '_' + fname

        if (skipexisting) and os.path.exists(tab_ofname):
            logging.info('skipping, products already exist %s' % f)
            if process_status_file is not None:
                update_process_status_file(process_status_file, f,
                                           ProcessStatus.EXISTING, -1, 0)

            continue
        args = (f, tab_ofname, mod_ofname, fig_prefix, config)
        kwargs = dict(fit_targetid=fit_targetid,
                      combine=combine,
                      mwonly=mwonly,
                      doplot=doplot,
                      minsn=minsn,
                      expid_range=expid_range,
                      overwrite=overwrite,
                      poolex=poolEx,
                      fitarm=fitarm,
                      cmdline=cmdline,
                      zbest_select=zbest_select,
                      process_status_file=process_status_file)
        proc_desi_wrapper(*args, **kwargs)

    if parallel:
        try:
            poolEx.shutdown(wait=True)
        except KeyboardInterrupt:
            for r in res:
                r.cancel()
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

    parser.add_argument('--output_tab_prefix',
                        help='Prefix of output table files',
                        type=str,
                        default='rvtab',
                        required=False)
    parser.add_argument('--output_mod_prefix',
                        help='Prefix of output model files',
                        type=str,
                        default='rvmod',
                        required=False)
    parser.add_argument('--fitarm',
                        help='Comma separated arms like b,r,z or b ',
                        type=str,
                        required=False)
    parser.add_argument('--figure_dir',
                        help='Prefix for the fit figures, i.e. fig_folder/',
                        type=str,
                        default='./')
    parser.add_argument('--figure_prefix',
                        help='Prefix for the fit figures, i.e. im',
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
    parser.add_argument('--process_status_file',
                        help='The name of the file where I put the names of' +
                        ' successfully processed files',
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument(
        '--overwrite',
        help='''If enabled the code will overwrite the existing products,
 otherwise it will attempt to update/append''',
        action='store_true',
        default=False)
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

    parser.add_argument('--doplot',
                        help='Make plots',
                        action='store_true',
                        default=False)

    parser.add_argument(
        '--combine',
        help='If enabled the code will simultaneously fit multiple spectra ' +
        'belonging to one targetid (OBSOLETE ???)',
        action='store_true',
        default=False)

    parser.add_argument('--allobjects',
                        help='Fit all objects not only MW_TARGET',
                        action='store_true',
                        default=False)

    args = parser.parse_args(args)

    if args.version:
        print(rvspecfit._version.version)
        sys.exit(0)

    log_level = args.log_level

    if args.log is not None:
        logging.basicConfig(filename=args.log, level=log_level)
    else:
        logging.basicConfig(level=log_level)

    input_files = args.input_files
    input_file_from = args.input_file_from
    output_dir, output_tab_prefix, output_mod_prefix = (args.output_dir,
                                                        args.output_tab_prefix,
                                                        args.output_mod_prefix)
    fig_prefix = args.figure_dir + '/' + args.figure_prefix
    nthreads = args.nthreads
    config_fname = args.config
    combine = args.combine
    mwonly = not args.allobjects
    doplot = args.doplot
    zbest_select = args.zbest_select
    minsn = args.minsn
    minexpid = args.minexpid
    maxexpid = args.maxexpid
    targetid_file_from = args.targetid_file_from
    targetid = args.targetid

    fitarm = args.fitarm
    if fitarm is not None:
        fitarm = fitarm.split(',')
    if input_files == [] and input_file_from is not None:
        raise Exception(
            '''You can only specify --input_files OR --input_file_from options
but not both of them simulatenously''')
    if input_files != []:
        files = input_files
    elif input_file_from is not None:
        files = []
        with open(input_file_from, 'r') as fp:
            for curl in fp:
                files.append(curl.rstrip())
    else:
        parser.print_help()
        raise RuntimeError('You need to specify the spectra you want to fit')

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

    proc_many(files,
              output_dir,
              output_tab_prefix,
              output_mod_prefix,
              fig_prefix,
              nthreads=nthreads,
              config_fname=config_fname,
              fit_targetid=fit_targetid,
              combine=combine,
              mwonly=mwonly,
              doplot=doplot,
              minsn=minsn,
              process_status_file=args.process_status_file,
              expid_range=(minexpid, maxexpid),
              skipexisting=args.skipexisting,
              overwrite=args.overwrite,
              fitarm=fitarm,
              cmdline=cmdline,
              zbest_select=zbest_select)


if __name__ == '__main__':
    main(sys.argv[1:])

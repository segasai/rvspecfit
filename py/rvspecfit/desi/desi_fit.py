import os
os.environ['OMP_NUM_THREADS'] = '1'
import glob
import warnings
import sys
import argparse
import time
import pandas
import itertools
import traceback
import importlib
import concurrent.futures
from collections import OrderedDict
import astropy.table as atpy
import astropy.io.fits as pyfits
import astropy.units as auni
import numpy as np

from rvspecfit import fitter_ccf, vel_fit, spec_fit, utils, spec_inter


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


def get_prim_header(versions={}):
    header = pyfits.Header()
    for i, (k, v) in enumerate(get_dep_versions().items()):
        header['DEPNAM%02d' % i] = k
        header['DEPVER%02d' % i] = v
    for i, (k, v) in enumerate(versions.items()):
        header['TMPLCON%d' % i] = k
        header['TMPLREV%d' % i] = v['revision']
        header['TMPLSVR%d' % i] = v['creation_soft_version']

    return header


def make_plot(specdata, yfit, title, fig_fname):
    """
    Make a plot with the spectra and fits

    Parameters:
    -----------
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

    alpha = 0.7
    line_width = 0.8
    dpi = 100
    plt.clf()
    ndat = len(specdata)
    plt.figure(1, figsize=(6, 2 * ndat), dpi=300)
    for i in range(ndat):
        plt.subplot(ndat, 1, i)
        plt.plot(specdata[i].lam, specdata[i].spec, 'k-', linewidth=line_width)
        plt.plot(specdata[i].lam,
                 yfit[i],
                 'r-',
                 alpha=alpha,
                 linewidth=line_width)
        if i == 0:
            plt.title(title)
        if i == ndat - 1:
            plt.xlabel(r'$\lambda$ [$\AA$]')
    plt.tight_layout()
    plt.savefig(fig_fname, dpi=dpi)


def valid_file(fname):
    """
    Check if all required extensions are present if yes return true
    """
    exts = pyfits.open(fname)
    extnames = [_.name for _ in exts]

    names0 = ['PRIMARY']
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
            print('WARNING Extensions %s are missing' % (','.join(missing)))
            return (False, True)
        return (True, True)
    return (True, False)


def proc_onespec(specdata,
                 setups,
                 config,
                 options,
                 fig_fname='fig.png',
                 doplot=True,
                 verbose=False):
    chisqs = {}
    chisqs_c = {}
    t1 = time.time()
    res = fitter_ccf.fit(specdata, config)
    t2 = time.time()
    paramDict0 = res['best_par']
    fixParam = []
    if res['best_vsini'] is not None:
        paramDict0['vsini'] = res['best_vsini']
    res1 = vel_fit.process(specdata,
                           paramDict0,
                           fixParam=fixParam,
                           config=config,
                           options=options,
                           verbose=verbose)
    t3 = time.time()
    chisq_cont_array = spec_fit.get_chisq_continuum(
        specdata, options=options)['chisq_array']
    t4 = time.time()
    outdict = dict(
        VRAD=res1['vel'] * auni.km / auni.s,
        VRAD_ERR=res1['vel_err'] * auni.km / auni.s,
        LOGG=res1['param']['logg'],
        TEFF=res1['param']['teff'] * auni.K,
        ALPHAFE=res1['param']['alpha'],
        FEH=res1['param']['feh'],
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
    rv_thresh = 5 * auni.km / auni.s
    rverr_thresh = 100 * auni.km / auni.s
    rvs_warn = 0
    bitmasks = {'CHISQ_WARN': 1, 'RV_WARN': 2, 'RVERR_WARN': 4}
    if ((outdict['CHISQ_TOT'] - outdict['CHISQ_C_TOT']) > chisq_thresh):
        rvs_warn |= bitmasks['CHISQ_WARN']
    if (np.abs(outdict['VRAD'] - config['max_vel'] * auni.km / auni.s) <
            rv_thresh) or (np.abs(outdict['VRAD'] - config['min_vel'] *
                                  auni.km / auni.s) < rv_thresh):
        rvs_warn |= bitmasks['RV_WARN']
    if (outdict['VRAD_ERR'] > rverr_thresh):
        rvs_warn |= bitmasks['RVERR_WARN']

    outdict['RVS_WARN'] = rvs_warn

    if doplot:
        title = 'logg=%.1f teff=%.1f [Fe/H]=%.1f [alpha/Fe]=%.1f Vrad=%.1f+/-%.1f' % (
            res1['param']['logg'], res1['param']['teff'], res1['param']['feh'],
            res1['param']['alpha'], res1['vel'], res1['vel_err'])
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
    if verbose:
        print('Timing1: ', t2 - t1, t3 - t2, t4 - t3)
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

    return sns


def read_data(fname, glued, setups):
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
                         minsn=None,
                         mwonly=True,
                         expid_range=None,
                         glued=False):
    """
    Identify fibers to fit 
    Currently that either uses MWS_TARGET or S/N cut
    Parameters:
    ----------
    fibermap: Table
        Fibermap table object
    sns: dict of numpy arrays 
        Array of S/Ns
    minsn: float
        Threshold S/N
    mwonly: bool
        Only fit MWS

    Returns:
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
        subset = subset & (fibermap["EXPID"] > mine) & (fibermap['EXPID'] <=
                                                        maxe)
    if minsn is not None:
        maxsn = np.max(np.array(list(sns.values())), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            subset = subset & (maxsn > minsn)
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
    large_error = 1e9
    sds = []

    for s in setups:
        spec = fluxes[s][seqid] * 1
        curivars = ivars[s][seqid] * 1
        medspec = np.nanmedian(spec)
        if medspec == 0:
            medspec = np.nanmedian(spec[spec > 0])
            if not np.isfinite(medspec):
                medspec = 1
        baddat = ~np.isfinite(spec + curivars)
        badmask = (curivars <= 0) | (masks[s][seqid] > 0) | baddat
        curivars[badmask] = medspec**2 / large_error**2
        spec[baddat] = medspec
        espec = 1. / curivars**.5
        sd = spec_fit.SpecData('desi_%s' % s,
                               waves[s],
                               spec,
                               espec,
                               badmask=badmask)
        sds.append(sd)
    return sds


def comment_filler(tab, desc):
    for i, k in enumerate(tab.data.columns):
        tab.header['TCOMM%d' % (i + 1)] = desc.get(k.name) or ''
    return tab


def put_empty_file(fname):
    pyfits.PrimaryHDU(header=get_prim_header()).writeto(fname,
                                                        overwrite=True,
                                                        checksum=True)


def proc_desi(fname,
              tab_ofname,
              mod_ofname,
              fig_prefix,
              config,
              fit_targetid,
              combine=False,
              mwonly=True,
              doplot=True,
              minsn=-1e9,
              verbose=False,
              expid_range=None,
              overwrite=False,
              poolex=None,
              fitarm=None):
    """
    Process One single file with desi spectra

    Parameters:
    -----------
    fname: str
        The filename with the spectra to be fitted
    ofname: str
        The filename where the table with parameters will be stored
    fig_prefix: str
        The prefix where the figures will be stored
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
    """

    options = {'npoly': 10}

    print('Processing', fname)
    valid, glued = valid_file(fname)
    if not valid:
        print('Not valid file:', fname)
        return 0

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
            print(
                'WARNING the size of the data in arm %s does not match the size of the fibermap; file %s; skipping...'
                % (_, fname))
            return 0
    targetid = fibermap['TARGETID']

    subset = select_fibers_to_fit(fibermap,
                                  sns,
                                  minsn=minsn,
                                  mwonly=mwonly,
                                  expid_range=expid_range,
                                  glued=glued)

    # skip if no need to fit anything
    if not (subset.any()):
        print('No fibers selected in file', fname)
        put_empty_file(tab_ofname)
        put_empty_file(mod_ofname)
        return 0

    # if we are combining

    # columns to include in the RVTAB
    columnsCopy = ['FIBER', 'REF_ID', 'TARGET_RA', 'TARGET_DEC', 'TARGETID']
    if not glued:
        columnsCopy.append('EXPID')

    #outdf = pandas.DataFrame()
    outdf = []

    # This will store best-fit model data
    models = {}
    for curs in setups:
        models['desi_%s' % curs] = []

    seqid_to_fit = get_unique_seqid_to_fit(targetid, subset, combine=combine)
    rets = []
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

        curbrick, curtargetid = curFiberRow['BRICKID'], curFiberRow['TARGETID']
        fig_fname = fig_prefix + '_%d_%d_%d.png' % (curbrick, curtargetid,
                                                    curseqid)

        rets.append((poolex.submit(
            proc_onespec, *(specdatas, setups, config, options),
            **dict(fig_fname=fig_fname, doplot=doplot,
                   verbose=verbose)), curFiberRow, curseqid))

    for r in rets:
        outdict, curmodel = r[0].result()
        versions = outdict['versions']
        del outdict['versions']  # I don't want to store it in the table
        curFiberRow, curseqid = r[1], r[2]

        for col in columnsCopy:
            outdict[col] = curFiberRow[col]
        for curs in setups:
            outdict['SN_%s' % curs.upper()] = sns[curs][curseqid]

        outdict['SUCCESS'] = True

        outdf.append(outdict)

        for ii, curd in enumerate(specdatas):
            models[curd.name].append(curmodel[ii])

    outdf1 = {}
    for k in outdf[0].keys():
        # we can't just concatenate quantities easily
        if isinstance(outdf[0][k], auni.Quantity):
            outdf1[k] = auni.Quantity([_[k] for _ in outdf])
        else:
            outdf1[k] = np.array([_[k] for _ in outdf])
    outtab = atpy.Table(outdf1)
    fibermap_subset_hdu = pyfits.BinTableHDU(atpy.Table(fibermap)[subset],
                                             name='FIBERMAP')
    outmod_hdus = [
        pyfits.PrimaryHDU(header=get_prim_header(versions=versions))
    ]

    # Column descriptions
    columnDesc = dict([('VRAD', 'Radial velocity'),
                       ('VRAD_ERR', 'Radial velocity error'),
                       ('VSINI', 'Stellar rotation velocity'),
                       ('LOGG', 'Log of surface gravity'),
                       ('TEFF', 'Effective temperature'),
                       ('FEH', '[Fe/H] from template fitting'),
                       ('ALPHAFE', '[alpha/Fe] from template fitting'),
                       ('CHISQ_TOT', 'Total chi-square for all arms'),
                       ('TARGETID', 'DESI targetid'),
                       ('EXPID', 'DESI exposure id'),
                       ('SUCCESS', "Did we succeed or fail"),
                       ('RVS_WARN', "RVSpecFit warning flag")])

    for curs in setups:
        curs = curs.upper()
        columnDesc['SN_%s' % curs] = ('Median S/N %s arm' % curs)
        columnDesc['CHISQ_%s' % curs] = ('Chi-square in the %s arm' % curs)
        columnDesc['CHISQ_C_%s' % curs] = (
            'Chi-square in the %s arm after fitting continuum only' % curs)

    # TODO
    # in the combine mode I don't know how to write the model
    #

    for curs in setups:
        outmod_hdus.append(
            pyfits.ImageHDU(waves[curs], name='%s_WAVELENGTH' % curs.upper()))
        outmod_hdus.append(
            pyfits.ImageHDU(np.vstack(models['desi_%s' % curs]),
                            name='%s_MODEL' % curs.upper()))
    outmod_hdus += [fibermap_subset_hdu]

    outtab_hdus = [
        pyfits.PrimaryHDU(header=get_prim_header(versions=versions)),
        comment_filler(pyfits.BinTableHDU(outtab, name='RVTAB'), columnDesc),
        fibermap_subset_hdu
    ]

    old_rvtab = None
    if os.path.exists(tab_ofname):
        try:
            old_rvtab = atpy.Table().read(tab_ofname,
                                          format='fits',
                                          hdu='FIBERMAP')
        except (FileNotFoundError, OSError) as e:
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
        refit_tab = atpy.Table(fibermap)[subset]
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
        #newids = np.array(merge['rowid_x'])[np.array(merge['_merge']=='left_only')]
        repset = np.array(merge['_merge'] == 'both_only')
        repid_old = np.array(merge['rowid_y'], dtype=int)[repset]
        #repid_new = np.array(merge['rowid_x'])[repset]

        keepmask = np.ones(len(old_rvtab), dtype=bool)
        keepmask[repid_old] = False
        # this is the subset of the old data that must
        # be kept
        merge_hdus(outmod_hdus, mod_ofname, keepmask, columnDesc, glued,
                   setups)
        merge_hdus(outtab_hdus, tab_ofname, keepmask, columnDesc, glued,
                   setups)
    return len(seqid_to_fit)


def merge_hdus(hdus, ofile, keepmask, columnDesc, glued, setups):
    allowed = ['FIBERMAP', 'RVTAB'
               ] + ['%s_MODEL' % _
                    for _ in setups] + ['%s_WAVELENGTH' % _ for _ in setups]

    for i in range(len(hdus)):
        if i == 0:
            continue
        curhdu = hdus[i]
        curhduname = curhdu.name

        if curhduname not in allowed:
            raise Exception('Weird exception', curhduname)
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
    # TODO protection against crash
    # write into temp file then rename
    pyfits.HDUList(hdus).writeto(ofile, overwrite=True, checksum=True)


def proc_desi_wrapper(*args, **kwargs):
    try:
        ret = proc_desi(*args, **kwargs)
    except Exception as e:
        print('failed with these arguments', args, kwargs)
        traceback.print_exc()
        raise


proc_desi_wrapper.__doc__ = proc_desi.__doc__


class FakeFuture:
    def __init__(self, x):
        self.x = x

    def result(self):
        return self.x


class FakeExecutor:
    def __init__(self):
        pass

    def submit(self, f, *args, **kw):
        return FakeFuture(f(*args, **kw))


def proc_many(files,
              output_dir,
              output_tab_prefix,
              output_mod_prefix,
              fig_prefix,
              config=None,
              nthreads=1,
              combine=False,
              targetid=None,
              mwonly=True,
              minsn=-1e9,
              doplot=True,
              verbose=False,
              expid_range=None,
              overwrite=False,
              skipexisting=False,
              fitarm=None):
    """
    Process many spectral files

    Parameters:
    -----------
    files: strings
        The files with spectra
    oprefix: string
        The prefix where the table with measurements will be stored
    fig_prefix: string
        The prfix where the figures will be stored
    combine: bool
        Fit spectra of same targetid together
    targetid: integer
        The targetid to fit (the rest will be ignored)
    mwonly: bool
        Only fit mws_target
    doplot: bool
        Plotting
    minsn: real
        THe min S/N to fit
    skipexisting: bool
        if True do not process anything if output files exist
    """
    config = utils.read_config(config)
    assert (config is not None)
    assert ('template_lib' in config)
    if nthreads > 1:
        parallel = True
    else:
        parallel = False

    if parallel:
        poolEx = concurrent.futures.ProcessPoolExecutor(nthreads)
    else:
        poolEx = FakeExecutor()
    res = []
    for f in files:
        fname = f.split('/')[-1]
        assert (len(f.split('/')) > 2)
        fdirs = f.split('/')
        folder_path = output_dir + '/' + fdirs[-3] + '/' + fdirs[-2] + '/'
        suffix = ('-'.join(fname.split('-')[1:]))
        os.makedirs(folder_path, exist_ok=True)
        tab_ofname = folder_path + output_tab_prefix + '-' + suffix
        mod_ofname = folder_path + output_mod_prefix + '-' + suffix

        if (skipexisting) and os.path.exists(tab_ofname):
            print('skipping, products already exist', f)
            continue
        args = (f, tab_ofname, mod_ofname, fig_prefix, config, targetid)
        kwargs = dict(combine=combine,
                      mwonly=mwonly,
                      doplot=doplot,
                      minsn=minsn,
                      verbose=verbose,
                      expid_range=expid_range,
                      overwrite=overwrite,
                      poolex=poolEx,
                      fitarm=fitarm)
        proc_desi(*args, **kwargs)

    if parallel:
        try:
            poolEx.shutdown(wait=True)
        except KeyboardInterrupt:
            for r in res:
                r.cancel()
            poolEx.shutdown(wait=False)
            raise


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--nthreads',
                        help='Number of threads for the fits',
                        type=int,
                        default=1)

    parser.add_argument('--config',
                        help='The filename of the configuration file',
                        type=str,
                        default=None)

    parser.add_argument('--input_files',
                        help='Space separated list of files to process',
                        type=str,
                        default=None,
                        nargs='+')
    parser.add_argument(
        '--input_file_from',
        help='Read the list of spectral files from the text file',
        type=str,
        default=None)

    parser.add_argument('--output_dir',
                        help='Output directory for the tables',
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument('--targetid',
                        help='Fit only a given targetid',
                        type=int,
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
                        help='Comma separated',
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

    parser.add_argument(
        '--overwrite',
        help=
        'If enabled the code will overwrite the existing products, otherwise it will attempt to update/append',
        action='store_true',
        default=False)

    parser.add_argument(
        '--skipexisting',
        help=
        'If enabled the code will completely skip if there are existing products',
        action='store_true',
        default=False)

    parser.add_argument('--doplot',
                        help='Make plots',
                        action='store_true',
                        default=False)

    parser.add_argument(
        '--combine',
        help=
        'If enabled the code will simultaneously fit multiple spectra belonging to one targetid',
        action='store_true',
        default=False)

    parser.add_argument('--verbose',
                        help='Verbose output',
                        action='store_true',
                        default=False)

    parser.add_argument('--allobjects',
                        help='Fit all objects not only MW_TARGET',
                        action='store_true',
                        default=False)

    args = parser.parse_args(args)
    input_files = args.input_files
    input_file_from = args.input_file_from
    verbose = args.verbose
    output_dir, output_tab_prefix, output_mod_prefix = args.output_dir, args.output_tab_prefix, args.output_mod_prefix
    fig_prefix = args.figure_dir + '/' + args.figure_prefix
    nthreads = args.nthreads
    config = args.config
    targetid = args.targetid
    combine = args.combine
    mwonly = not args.allobjects
    doplot = args.doplot
    minsn = args.minsn
    minexpid = args.minexpid
    maxexpid = args.maxexpid
    fitarm = args.fitarm
    if fitarm is not None:
        fitarm = fitarm.split(',')
    if input_files is not None and input_file_from is not None:
        raise Exception(
            'You can only specify --input_files OR --input_file_from options but not both of them simulatenously'
        )

    if input_files is not None:
        files = input_files
    elif input_file_from is not None:
        files = []
        with open(input_file_from, 'r') as fp:
            for l in fp:
                files.append(l.rstrip())
    else:
        raise Exception('You need to specify the spectra you want to fit')

    proc_many(files,
              output_dir,
              output_tab_prefix,
              output_mod_prefix,
              fig_prefix,
              nthreads=nthreads,
              config=config,
              targetid=targetid,
              combine=combine,
              mwonly=mwonly,
              doplot=doplot,
              minsn=minsn,
              verbose=verbose,
              expid_range=(minexpid, maxexpid),
              skipexisting=args.skipexisting,
              overwrite=args.overwrite,
              fitarm=fitarm)


if __name__ == '__main__':
    main(sys.argv[1:])

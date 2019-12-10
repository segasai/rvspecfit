import os
os.environ['OMP_NUM_THREADS'] = '1'
import glob
import sys
import argparse
import time
import pandas
import itertools
import traceback
import concurrent.futures
from collections import OrderedDict
import astropy.table as atpy
import astropy.io.fits as pyfits
import numpy as np

from rvspecfit import fitter_ccf, vel_fit, spec_fit, utils


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
    plt.figure(1, figsize=(6, 6), dpi=300)
    plt.subplot(3, 1, 1)
    plt.plot(specdata[0].lam, specdata[0].spec, 'k-', linewidth=line_width)
    plt.plot(
        specdata[0].lam,
        yfit[0],
        'r-',
        alpha=alpha,
        linewidth=line_width)
    plt.title(title)
    plt.subplot(3, 1, 2)
    plt.plot(specdata[1].lam, specdata[1].spec, 'k-', linewidth=line_width)
    plt.plot(
        specdata[1].lam,
        yfit[1],
        'r-',
        alpha=alpha,
        linewidth=line_width)
    plt.subplot(3, 1, 3)
    plt.plot(specdata[2].lam, specdata[2].spec, 'k-', linewidth=line_width)
    plt.plot(
        specdata[2].lam,
        yfit[2],
        'r-',
        alpha=alpha,
        linewidth=line_width)
    plt.xlabel(r'$\lambda$ [$\AA$]')
    plt.tight_layout()
    plt.savefig(fig_fname, dpi=dpi)


def valid_file(fname):
    """
    Check if all required extensions are present if yes return true
    """
    exts = pyfits.open(fname)
    extnames = [_.name for _ in exts]

    arms = 'B', 'R', 'Z'
    prefs = 'WAVELENGTH', 'FLUX', 'IVAR', 'MASK'
    names0 = ['PRIMARY']
    reqnames = names0 + [
        '%s_%s' % (_, __) for _, __ in itertools.product(arms, prefs)
    ]
    missing = []
    for curn in reqnames:
        if curn not in extnames:
            missing.append(curn)
    if len(missing) != 0:
        print('WARNING Extensions %s are missing' % (','.join(missing)))
        return False
    return True

def proc_onespec(specdata, setups, config, options, fig_fname_mask,
                 doplot=True):
    chisqs = {}
    chisqs_c  = {} 
    t1 = time.time()
    res = fitter_ccf.fit(specdata, config)
    t2 = time.time()
    paramDict0 = res['best_par']
    fixParam = []
    if res['best_vsini'] is not None:
        paramDict0['vsini'] = res['best_vsini']
    res1 = vel_fit.process(
        specdata,
        paramDict0,
        fixParam=fixParam,
        config=config,
        options=options)
    t3 = time.time()
    chisq_cont_array = spec_fit.get_chisq_continuum(
                specdata, options=options)
    t4 = time.time()
    outdict= dict(
                  vrad=res1['vel'],
                  vrad_err=res1['vel_err'],
                  logg=res1['param']['logg'],
                  teff=res1['param']['teff'],
                  alpha=res1['param']['alpha'],
                  feh=res1['param']['feh'],
                  vsini=res1['vsini'],
                  nexp=len(specdata)/len(setups),
                  )

    for i, curd in enumerate(specdata):
        if curd.name not in chisqs:
            chisqs[curd.name]=0
            chisqs_c[curd.name]=0
        chisqs[curd.name]+=res1['chisq_array'][i]
        chisqs_c[curd.name]+=chisq_cont_array[i]

    for s in chisqs.keys():
        outdict['chisq_tot']=sum(chisqs.values())
        outdict['chisq_%s' % s.replace('desi_','')]=chisqs[s]
        outdict['chisq_c_%s' % s.replace('desi_','')]=float(chisqs_c[s])

    if doplot:
        title = 'logg=%.1f teff=%.1f [Fe/H]=%.1f [alpha/Fe]=%.1f Vrad=%.1f+/-%.1f' % (
        res1['param']['logg'], res1['param']['teff'], res1['param']['feh'],
        res1['param']['alpha'], res1['vel'], res1['vel_err'])
        if len(specdata)>len(setups):
            for i in range(len(specdata)//len(setups)):
                sl = slice(i*len(setups),(i+1)*len(setups))
                make_plot(specdata[sl], 
                  res1['yfit'][sl], title, fig_fname_mask%i)
        else:
            make_plot(specdata,
                  res1['yfit'], title, fig_fname_mask)

    return outdict, res1['yfit']

def proc_desi(fname, tab_ofname, mod_ofname, fig_prefix, config, fit_targetid, combine=False,
              mwonly=True, doplot=True, minsn=-1e9):
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
    """

    options = {'npoly': 10}

    print('Processing', fname)
    if not valid_file(fname):
        return
    tab = pyfits.getdata(fname, 'FIBERMAP')
    if mwonly:
        mws = tab['MWS_TARGET']!=0
    else:
        mws = np.ones(len(tab), dtype=bool)
    if not (mws.any()):
        return
    targetid = tab['TARGETID']
    brickid = tab['BRICKID']
    columnsCopy = ['FIBER', 'REF_ID','TARGET_RA','TARGET_DEC']
    seqid = np.arange(len(targetid))
    fiberSubset = np.zeros(len(tab),dtype=bool)

    setups = ('b', 'r', 'z')
    fluxes = {}
    ivars = {}
    waves = {}
    masks = {}
    for s in setups:
        fluxes[s] = pyfits.getdata(fname, '%s_FLUX' % s.upper())
        ivars[s] = pyfits.getdata(fname, '%s_IVAR' % s.upper())
        masks[s] = pyfits.getdata(fname, '%s_MASK' % s.upper())
        waves[s] = pyfits.getdata(fname, '%s_WAVELENGTH' % s.upper())

    large_error = 1e9
    
    utargetid, uuid  = np.unique(targetid[mws],return_index=True)
    uuid = np.nonzero(mws)[0][uuid]
    if not combine:
        uuid = seqid

    outdf = pandas.DataFrame()
    models = {}
    for curs in setups:
        models['desi_%s'%curs]=[]
    
    for curseqid in uuid:
        curtargetid = targetid[curseqid]

        specdata = []

        if fit_targetid is not None and curtargetid != fit_targetid:
            continue
        if combine:
            xids = np.nonzero(targetid==curtargetid)[0]
        else:
            xids = [curseqid]

        specdatas = []
        sns = [] # sns of all the datasets collected
        
        # collect data (if combining that means multiple spectra)
        for curid in xids:
            curbrick = brickid[curid]
            curCols = dict([(_,tab[_][curid]) for _ in columnsCopy])
            specdata = []
            cursn = {}
            for s in setups:
                spec = fluxes[s][curid]
                curivars = ivars[s][curid]
                badmask = (curivars <= 0) | (masks[s][curid] > 0)
                curivars[badmask] = 1. / large_error**2
                espec = 1. / curivars**.5
                cursn[s] = np.nanmedian(spec / espec)
                specdata.append(
                    spec_fit.SpecData(
                    'desi_%s' % s, waves[s], spec, espec, badmask=badmask))
            sns.append(cursn)
            specdatas.append(specdata)
        fig_fname_mask = fig_prefix + '_%d_%d_%%d.png' % (curbrick, curtargetid)

        curmaxsn = -1
        for i,specdata in enumerate(specdatas):
            for f in setups:
                curmaxsn = max(sns[i][f],curmaxsn)
        if curmaxsn < minsn:
            continue
        if combine:
            specdata = sum(specdatas,[])
            curmask = fig_fname_mask
            if len(specdata)==len(setups):
                curmask=curmask%0
            outdict,curmodel = proc_onespec(specdata, setups, config, options, curmask, doplot=doplot)
            outdict['BRICKID']=curbrick
            outdict['TARGETID']=curtargetid
            for col in curCols.keys():
                outdict[col] = curCols[col]
            for f in setups:
                outdict['sn_%s'%f] = np.nanmedian([_[f] for _ in sns])
            outdf =  outdf.append(pandas.DataFrame([outdict]), True)
        else:
            assert(len(specdatas)==1)
            outdict,curmodel = proc_onespec(specdata, setups, config, options, fig_fname_mask%i, doplot=doplot)
            outdict['BRICKID']=curbrick
            outdict['TARGETID']=curtargetid
            for col in curCols.keys():
                outdict[col] = curCols[col]
                
            for f in setups:
                outdict['sn_%s'%f] = sns[0][f]

            outdf =  outdf.append(pandas.DataFrame([outdict]), True)
            for ii, curd in enumerate(specdata):
                models[curd.name].append(curmodel[i])
            
        fiberSubset[curseqid] = True
    if len(outdf)==0:
        return
    fibermap_copy = pyfits.BinTableHDU(atpy.Table(tab)[fiberSubset],name='FIBERMAP')
    outputmod = [pyfits.PrimaryHDU()]

    # TODO 
    # in the combine mode I don't know how to write the model
    #

    for curs in setups:
        outputmod.append(pyfits.ImageHDU(pyfits.getdata(fname, '%s_WAVELENGTH' % curs.upper()),
                        name ='%s_WAVELENGTH'%curs.upper()))
        outputmod.append(pyfits.ImageHDU(np.vstack(models['desi_%s'%curs]),
                                         name='%s_MODEL'%curs.upper()))
    pyfits.HDUList(outputmod+[fibermap_copy]).writeto(mod_ofname, overwrite=True)

    outtab = atpy.Table.from_pandas(outdf)
    hdulist = pyfits.HDUList([pyfits.PrimaryHDU(),pyfits.BinTableHDU(outtab),
                              fibermap_copy])
    hdulist.writeto(tab_ofname, overwrite=True)
    return 1;

def proc_desi_wrapper(*args, **kwargs):
    try:
        ret = proc_desi(*args, **kwargs)
    except Exception as e:
        print('failed with these arguments', args, kwargs)
        traceback.print_exc()
        raise


proc_desi_wrapper.__doc__ = proc_desi.__doc__


def proc_many(files,
              output_dir,
              output_tab_prefix,
              output_mod_prefix,
              fig_prefix,
              config=None,
              nthreads=1,
              overwrite=True,
              combine=False,
              targetid=None,
              mwonly=True,
              minsn=-1e9,
              doplot=True):
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
    """
    config = utils.read_config(config)

    if nthreads > 1:
        parallel = True
    else:
        parallel = False

    if parallel:
        poolEx = concurrent.futures.ProcessPoolExecutor(nthreads)
    res = []
    for f in files:
        fname = f.split('/')[-1]
        assert(len(f.split('/'))>2)
        fdirs = f.split('/')
        folder_path = output_dir + '/' + fdirs[-3] + '/' + fdirs[-2] + '/'
        suffix =  ('-'.join(fname.split('-')[1:]))
        os.makedirs(folder_path, exist_ok=True)
        tab_ofname = folder_path + output_tab_prefix + '-'+suffix
        mod_ofname = folder_path + output_mod_prefix + '-'+suffix

        if (not overwrite) and os.path.exists(tab_ofname):
            print('skipping, products already exist', f)
            continue
        args = (f, tab_ofname, mod_ofname, fig_prefix, config, targetid)
        kwargs = dict(combine=combine,
                      mwonly=mwonly,
                      doplot=doplot,
                      minsn=minsn)
        if parallel:
            res.append(
                poolEx.submit(proc_desi_wrapper, 
                            *args, **kwargs)
            )
        else:
            proc_desi_wrapper(*args, **kwargs)
    
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

    parser.add_argument(
        '--nthreads',
        help='Number of threads for the fits',
        type=int,
        default=1)

    parser.add_argument(
        '--config',
        help='The filename of the configuration file',
        type=str,
        default=None)

    parser.add_argument(
        '--input_files',
        help='Space separated list of files to process',
        type=str,
        default=None,
        nargs='+')
    parser.add_argument(
        '--input_file_from',
        help='Read the list of spectral files from the text file',
        type=str,
        default=None)

    parser.add_argument(
        '--output_dir',
        help='Output directory for the tables',
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        '--targetid',
        help='Fit only a given targetid',
        type=int,
        default=None,
        required=False)
    parser.add_argument(
        '--minsn',
        help='Fit only S/N larger than this',
        type=float,
        default=-1e9,
        required=False)
    parser.add_argument(
        '--output_tab_prefix',
        help='Prefix of output table files',
        type=str,
        default='rvtab',
        required=False)
    parser.add_argument(
        '--output_mod_prefix',
        help='Prefix of output model files',
        type=str,
        default='rvmod',
        required=False)

    parser.add_argument(
        '--figure_dir',
        help='Prefix for the fit figures, i.e. fig_folder/',
        type=str,
        default='./')
    parser.add_argument(
        '--figure_prefix',
        help='Prefix for the fit figures, i.e. im',
        type=str,
        default='fig',
        required=False)

    parser.add_argument(
        '--overwrite',
        help=
        'If enabled the code will overwrite the existing products, otherwise it will skip them',
        action='store_true',
        default=False)

    parser.add_argument(
        '--doplot',
        help=
        'If enabled the code will overwrite the existing products, otherwise it will skip them',
        action='store_true',
        default=False)

    parser.add_argument(
        '--combine',
        help=
        'If enabled the code will simultaneously fit multiple spectra belonging to one targetid',
        action='store_true',
        default=False)

    parser.add_argument(
        '--allobjects',
        help=
        'Fit all objects not only MW_TARGET',
        action='store_true',
        default=False)

    args = parser.parse_args(args)
    input_files = args.input_files
    input_file_from = args.input_file_from

    output_dir,output_tab_prefix,output_mod_prefix = args.output_dir, args.output_tab_prefix, args.output_mod_prefix
    fig_prefix = args.figure_dir + '/' + args.figure_prefix
    nthreads = args.nthreads
    config = args.config
    targetid = args.targetid
    combine = args.combine
    mwonly = not args.allobjects
    doplot = args.doplot
    minsn = args.minsn
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

    proc_many(
        files,
        output_dir,
        output_tab_prefix,
        output_mod_prefix,
        fig_prefix,
        nthreads=nthreads,
        overwrite=args.overwrite,
        config=config,
        targetid=targetid,
        combine=combine,
        mwonly=mwonly,
        doplot=doplot,
        minsn=minsn)


if __name__ == '__main__':
    main(sys.argv[1:])

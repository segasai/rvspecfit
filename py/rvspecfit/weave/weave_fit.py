import glob
import sys
import os
import argparse
import time
import itertools
import multiprocessing as mp
from collections import OrderedDict

import matplotlib
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import astropy.table as atpy

from rvspecfit import fitter_ccf, vel_fit, spec_fit, utils

def make_plot(specdata, res_dict, title, fig_fname):
    """
    Make a plot with the spectra and fits

    Parameters:
    -----------
    specdata: SpecData object
        The object with specdata
    res_dict: list
        The list of dictionaries with fit results. The dictionaries must have yfit key
    title: string
        The figure title
    fig_fname: string
        The filename of the figure
    """
    alpha = 0.7
    line_width = 0.8
    plt.clf()
    plt.figure(1, figsize=(6, 6), dpi=300)
    l = len(specdata[0].lam)
    l2 = l//2
    plt.subplot(4, 1, 1)
    plt.plot(specdata[0].lam[:l2], specdata[0].spec[:l2], 'k-', linewidth=line_width)
    plt.plot(specdata[0].lam[:l2], res_dict['yfit'][0][:l2], 'r-',
             alpha=alpha, linewidth=line_width)
    plt.title(title)

    plt.subplot(4, 1, 2)
    plt.plot(specdata[0].lam[l2:], specdata[0].spec[l2:], 'k-', linewidth=line_width)
    plt.plot(specdata[0].lam[l2:], res_dict['yfit'][0][l2:], 'r-',
             alpha=alpha, linewidth=line_width)   
    l = len(specdata[1].lam)
    l2 = l//2

    plt.subplot(4, 1, 3)
    plt.plot(specdata[1].lam[:l2], specdata[1].spec[:l2], 'k-', linewidth=line_width)
    plt.plot(specdata[1].lam[:l2], res_dict['yfit'][1][:l2], 'r-',
             alpha=alpha, linewidth=line_width)
    plt.title(title)
    plt.subplot(4, 1, 4)
    plt.plot(specdata[1].lam[l2:], specdata[1].spec[l2:], 'k-', linewidth=line_width)
    plt.plot(specdata[1].lam[l2:], res_dict['yfit'][1][l2:], 'r-',
             alpha=alpha, linewidth=line_width)
    plt.xlabel(r'$\lambda$ [$\AA$]')
    plt.tight_layout()
    plt.savefig(fig_fname)

def valid_file(fname):
    """
    Check if all required extensions are present if yes return true
    """
    exts = pyfits.open(fname)
    extnames = [_.name for _ in exts]


    #arms = 'B','R','Z'
    #prefs = 'WAVELENGTH', 'FLUX', 'IVAR', 'MASK'
    names0 = ['RED_DATA', 'RED_IVAR','FIBTABLE']
    reqnames = names0
    missing = []
    for curn in reqnames:
        if curn not in extnames:
            missing.append(curn)
    if len(missing)!=0:
        print ('WARNING Extensions %s are missing'%(','.join(missing)))
        return False
    return True

def proc_weave(fnames, fig_prefix, config, threadid, nthreads):
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
    """

    options = {'npoly': 15}

    print('Processing', fnames)
    fnames = fnames.split(',')
    if not valid_file(fnames[0]):
        return
    tab = pyfits.getdata(fnames[0], 'FIBTABLE')
    hdr = pyfits.getheader(fnames[0])
    #mws = tab['MWS_TARGET']
    targetid = tab['TARGID']
    brick_name = hdr['OBID'].replace('.','').replace('/','').replace('_','')
    #xids = np.nonzero(mws)[0]
    #setups = ('b', 'r', 'z')
    fluxes = {}
    ivars = {}
    waves = {}
    masks = {}
    setups = ('b','r')

    xids= np.nonzero( 
        ( tab['TARGCAT']=='GA_LRhighlat') | 
        ( tab['TARGCAT']=='GA_LRdisk') 
    )[0]
    if len(xids)>0:
        tids = np.arange(len(xids)) // max(len(xids)//nthreads,1)
        xids  = xids[ tids==threadid]
    if len(xids)==0:
        return None

    arms = [pyfits.getheader(f)['CAMERA'].replace('WEAVE','') for f in fnames]
    if arms == ['RED','BLUE'] or arms==['BLUE','RED']:
        if arms==['RED','BLUE']:
            fnames = fnames[::-1]
    else:
        raise Exception('No RED/BLUE setups')


    for fname,s in zip(fnames,setups):
        curarm = {'b':'BLUE', 'r':'RED'}[s]
        fluxes[s] = pyfits.getdata(fname, '%s_DATA'%curarm)
        ivars[s] = pyfits.getdata(fname, '%s_IVAR'%curarm)
        masks[s] = (ivars[s]==0).astype(int)
        pix = np.arange(fluxes[s].shape[1])
        wc = pywcs.WCS(pyfits.getheader(fname,'%s_DATA'%curarm));
        waves[s] = wc.all_pix2world(np.array([pix,pix*0]).T,0).T[0] * 1e10
        masks[s] = masks[s] | ((waves[s]>=8130) & (waves[s]<8350))
        masks[s] = masks[s] | ((waves[s]>=6850) & (waves[s]<7000))
        masks[s] = masks[s] | ((waves[s]>=8940) & (waves[s]<9240))
        masks[s] = masks[s] | ((waves[s]>=9250) & (waves[s]<9545))
        masks[s] = masks[s] | ((waves[s]>=9550) & (waves[s]<10000))

    columns = ['brickname','target_id',
               'vrad','vrad_err',
               'logg','teff',
               'vsini','feh','alpha',
               'chisq_tot']
    for s in setups:
        columns.append('sn_%s'%s)
        columns.append('chisq_%s'%s)
        columns.append('chisq_c_%s'%s)

    outdict = OrderedDict()
    for c in columns:
        outdict[c]=[]
    large_error = 1e9
    for curid in xids:
        specdata = []
        curbrick = brick_name
        curtargetid = targetid[curid].replace('"','')
        fig_fname = fig_prefix + '_%s_%s.png' % (curbrick, curtargetid)
        sns = {}
        chisqs = {}
        for s in setups:
            spec = fluxes[s][curid]
            curivars = ivars[s][curid]
            badmask = ( curivars <= 0 ) | (masks[s][curid] > 0)
            curivars[badmask] = 1. / large_error**2
            espec = 1. / curivars**.5
            sns[s] = np.nanmedian (spec/espec)
            specdata.append(
                spec_fit.SpecData('weave_%s' % s,
                                  waves[s], spec, espec,
                                  badmask = badmask))
        t1 = time.time()
        res = fitter_ccf.fit(specdata, config)
        t2 = time.time()
        paramDict0 = res['best_par']
        fixParam = []
        if res['best_vsini'] is not None:
            paramDict0['vsini'] = res['best_vsini']
        res1 = vel_fit.process(specdata, paramDict0, fixParam=fixParam,
                            config=config, options=options)
        t3 = time.time()
        chisq_cont_array = spec_fit.get_chisq_continuum( specdata, options=options)
        t4 = time.time()
        outdict['brickname'].append(curbrick)
        outdict['target_id'].append(curtargetid)
        outdict['vrad'].append(res1['vel'])
        outdict['vrad_err'].append(res1['vel_err'])
        outdict['logg'].append(res1['param']['logg'])
        outdict['teff'].append(res1['param']['teff'])
        outdict['alpha'].append(res1['param']['alpha'])
        outdict['feh'].append(res1['param']['feh'])
        outdict['chisq_tot'].append(sum(res1['chisq_array']))
        for i, s in enumerate(setups):
            outdict['chisq_%s'%s].append(res1['chisq_array'][i])
            outdict['chisq_c_%s'%s].append(float(chisq_cont_array[i]))
            outdict['sn_%s'%(s,)].append(sns[s])

        outdict['vsini'].append(res1['vsini'])

        title = 'logg=%.1f teff=%.1f [Fe/H]=%.1f [alpha/Fe]=%.1f Vrad=%.1f+/-%.1f' % (res1['param']['logg'],
                                                                                      res1['param']['teff'],
                                                                                      res1['param']['feh'],
                                                                                      res1['param']['alpha'],
                                                                                      res1['vel'],
                                                                                      res1['vel_err'])
        make_plot(specdata, res1, title, fig_fname)
    outtab = atpy.Table(outdict)
    return outtab

def proc_weave_wrapper(*args, **kwargs):
    try:
        ret = proc_weave(*args, **kwargs)
        return ret
    except:
        print('failed with these arguments', args, kwargs)
        raise


proc_weave_wrapper.__doc__ = proc_weave.__doc__

def proc_many(files, oprefix, fig_prefix, config=None, nthreads=1, overwrite=True):
    """
    Process many spectral files

    Parameters:
    -----------
    mask: string
        The filename mask with spectra, i.e path/*fits
    oprefix: string
        The prefix where the table with measurements will be stored
    fig_prefix: string
        The prfix where the figures will be stored
    """
    config = utils.read_config(config)

    if nthreads > 1:
        parallel = True
    else:
        parallel = False

    if parallel:
        pool = mp.Pool(nthreads)
    res = []
    for f in files:
        fname = f.split('/')[-1]
        ofname = oprefix + 'outtab_' + fname
        if (not overwrite) and os.path.exists(ofname):
            print('skipping, products already exist', f)
            continue
        if parallel:
            for i in range(nthreads):
                res.append(pool.apply_async(
                proc_weave_wrapper, (f, fig_prefix, config,i, nthreads)))
            tabs = []
            for r in res:
                tabs.append(r.get())
                
            tabs = ([_ for _ in tabs if _ is not None] )
            if len(tabs)==0:
                continue
            tabs = atpy.vstack(tabs)    
            tabs.write(ofname, overwrite=True)

        else:
            
            tabs=proc_weave_wrapper(f, fig_prefix, config, 0, 1)
            if tabs is not None:
                tabs.write(ofname, overwrite=True)

    if parallel:
        pool.close()
        pool.join()

def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--nthreads', help='Number of threads for the fits',
                        type=int, default=1)

    parser.add_argument('--config',
                        help='The filename of the configuration file',
                        type=str, default=None)

    parser.add_argument('--input_file_mask',
                        help='The file mask of spectra, i.e. spectra*fits',
                        type=str, default=None)
    parser.add_argument('--input_file',
                        help='Read the list of spectra from the file',
                        type=str, default=None)

    parser.add_argument('--output_dir',
                        help='Output directory for the tables',
                        type=str, default=None, required=True)
    parser.add_argument('--output_tab_prefix',
                        help='Prefix of output table files',
                        type=str, default='outtab', required=False)

    parser.add_argument('--figure_dir',
                        help='Prefix for the fit figures, i.e. fig_folder/',
                        type=str, default='./')
    parser.add_argument('--figure_prefix',
                        help='Prefix for the fit figures, i.e. im',
                        type=str, default='fig',required=False)

    parser.add_argument('--overwrite',
                        help='If enabled the code will overwrite the existing products, otherwise it will skip them',
                        action='store_true', default=False)

    args = parser.parse_args(args)
    mask = args.input_file_mask
    input_file = args.input_file

    oprefix = args.output_dir+'/'+args.output_tab_prefix
    fig_prefix = args.figure_dir + '/' + args.figure_prefix
    nthreads = args.nthreads
    config = args.config

    if mask is not None and input_file is not None:
        raise Exception('You can only specify --mask OR --input_file options but not both of them simulatenously')

    if mask is not None:
        files = glob.glob(mask)
    elif input_file is not None:
        files = []
        with open(input_file,'r') as fp:
            for l in fp:
                files.append(l.rstrip())
    else:
        raise Exception('You need to specify the spectra you want to fit')

    proc_many(files, oprefix, fig_prefix, nthreads=nthreads,
           overwrite=args.overwrite, config=config)

if __name__ == '__main__':
    main(sys.argv[1:])

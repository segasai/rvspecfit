import glob
import sys
import os
import matplotlib
import argparse
import multiprocessing as mp
import astropy.io.fits as pyfits
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import astropy.table

sys.path.append('../')
import fitter_ccf
import vel_fit
import spec_fit

import utils

def procdesi(fname, ofname, fig_prefix, config):
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

    print('Processing', fname)
    tab = pyfits.getdata(fname, 'FIBERMAP')
    mws = tab['MWS_TARGET']
    targetid = tab['TARGETID']
    brick_name = tab['BRICKNAME']
    xids = np.nonzero(mws)[0]
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

    outdict = {'brickname': [],
               'target_id': [],
               'vrad': [],
               'vrad_err': [],
               'logg': [],
               'teff': [],
               'vsini': [],
               'feh': [],
               'chisq': []}
    large_error = 1e9
    for curid in xids:
        specdata = []
        curbrick = brick_name[curid]
        curtargetid = targetid[curid]
        fig_fname = fig_prefix + '_%s_%d.png' % (curbrick, curtargetid)
        for s in setups:
            spec = fluxes[s][curid]
            curivars = ivars[s][curid]
            badmask = ( curivars <= 0 ) | (masks[s][curid] > 0)
            curivars[badmask] = 1. / large_error**2
            espec = 1. / curivars**.5
            specdata.append(
                spec_fit.SpecData('desi_%s' % s,
                                  waves[s], spec, espec,
                                  badmask = badmask))
        options = {'npoly': 15}
        res = fitter_ccf.fit(specdata, config)
        paramDict0 = res['best_par']
        fixParam = []
        if res['best_vsini'] is not None:
            paramDict0['vsini'] = res['best_vsini']
        res1 = vel_fit.doit(specdata, paramDict0, fixParam=fixParam,
                            config=config, options=options)
        outdict['brickname'].append(curbrick)
        outdict['target_id'].append(curtargetid)
        outdict['vrad'].append(res1['vel'])
        outdict['vrad_err'].append(res1['vel_err'])
        outdict['logg'].append(res1['param']['logg'])
        outdict['teff'].append(res1['param']['teff'])
        outdict['feh'].append(res1['param']['feh'])
        outdict['chisq'].append(res1['chisq'])
        outdict['vsini'].append(res1['vsini'])
        title = 'logg=%.1f teff=%.1f [Fe/H]=%.1f [alpha/Fe]=%.1f Vrad=%.1f+/-%.1f' % (res1['param']['logg'],
                                                                                      res1['param']['teff'],
                                                                                      res1['param']['feh'],
                                                                                      res1['param']['alpha'],
                                                                                      res1['vel'],
                                                                                      res1['vel_err'])
        alpha = 0.5
        line_width = 0.8
        plt.clf()
        plt.figure(1, figsize=(6, 6), dpi=300)
        plt.subplot(3, 1, 1)
        plt.plot(specdata[0].lam, specdata[0].spec, 'k-', linewidth=line_width)
        plt.plot(specdata[0].lam, res1['yfit'][0], 'r-',
                 alpha=alpha, linewidth=line_width)
        plt.title(title)
        plt.subplot(3, 1, 2)
        plt.plot(specdata[1].lam, specdata[1].spec, 'k-', linewidth=line_width)
        plt.plot(specdata[1].lam, res1['yfit'][1], 'r-',
                 alpha=alpha, linewidth=line_width)
        plt.subplot(3, 1, 3)
        plt.plot(specdata[2].lam, specdata[2].spec, 'k-', linewidth=line_width)
        plt.plot(specdata[2].lam, res1['yfit'][2], 'r-',
                 alpha=alpha, linewidth=line_width)
        plt.xlabel(r'$\lambda$ [$\AA$]')
        plt.tight_layout()
        plt.savefig(fig_fname)
    outtab = astropy.table.Table(outdict)
    outtab.write(ofname, overwrite=True)


def procdesiWrapper(*args, **kwargs):
    try:
        ret = procdesi(*args, **kwargs)
    except:
        print('failed with these arguments', args, kwargs)
        raise


procdesiWrapper.__doc__ = procdesi.__doc__

def domany(files, oprefix, fig_prefix, config=None, nthreads=1, overwrite=True):
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
            res.append(pool.apply_async(
                procdesiWrapper, (f, ofname, fig_prefix, config)))
        else:
            procdesiWrapper(f, ofname, fig_prefix, config)
    if parallel:
        for r in res:
            r.get()
        pool.close()
        pool.join()


if __name__ == '__main__':
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
                        type=str, default='outtab')
    parser.add_argument('--fig_prefix',
                        help='Prefix for the fit figures, i.e. fig_folder/im',
                        type=str, default='./')
    parser.add_argument('--overwrite', 
                        help='If enabled the code will overwrite the existing products, otherwise it will skip them',
                        action='store_true', default=False)

    args = parser.parse_args()
    mask = args.input_file_mask
    input_file = args.input_file

    oprefix = args.output_dir+'/'+args.output_tab_prefix
    fig_prefix = args.fig_prefix
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

    domany(files, oprefix, fig_prefix, nthreads=nthreads,
           overwrite=args.overwrite, config=config)


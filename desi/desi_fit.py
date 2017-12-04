import glob
import sys
import matplotlib
matplotlib.use('Agg')
sys.path.append('../')
import vel_fit
import spec_fit
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import astropy.table
import fitter_ccf
import utils
import multiprocessing as mp

config = utils.read_config()



# read data

def procdesi(fname, ofname, fig_prefix):
    tab = pyfits.getdata(fname, 'FIBERMAP')
    mws = tab['MWS_TARGET']
    targetid = tab['TARGETID']
    brick_name = tab['BRICKNAME']
    xids  = np.nonzero(mws)[0]
    setups = ('b','r','z')
    fluxes = {} 
    ivars = {} 
    waves = {}
    for s in setups:
        fluxes[s] = pyfits.getdata(fname, '%s_FLUX'%s.upper())
        ivars[s] = pyfits.getdata(fname, '%s_IVAR'%s.upper())
        waves[s] = pyfits.getdata(fname, '%s_WAVELENGTH'%s.upper())

    outtab= astropy.table.Table()
    outdict = {'brickname':[],
               'target_id':[],
               'vrad':[],
               'vrad_err':[],
               'logg':[],
               'teff':[],
               'vsini':[],
               'feh':[],
               'chisq':[]}

    for curid in xids:
        specdata = []
        curbrick = brick_name[curid]
        curtargetid = targetid[curid]
        for s in setups:
            specdata.append(
                spec_fit.SpecData('desi_%s'%s, 
                                  waves[s],
                                  fluxes[s][curid],
                                  1./(ivars[s][curid])**.5))
        options = {'npoly': 15}
        res = fitter_ccf.fit(specdata, config)
        paramDict0 = res['bestpar']
        fixParam = []
        if res['bestvsini'] is not None:
            paramDict0['vsini'] = res['bestvsini']
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
        plt.clf()
        plt.figure(1,figsize=(6, 2), dpi=300)
        plt.plot(specdata[0].lam, specdata[0].spec, 'k-')
        plt.plot(specdata[0].lam, res1['yfit'][0], 'r-')
        plt.tight_layout()
        plt.savefig(fig_prefix+'_%s_%d.png'%(curbrick, curtargetid))
    outtab= astropy.table.Table(outdict)
    outtab.write(ofname)

def domany(mask, oprefix, fig_prefix):
    fs= glob.glob(mask)
    pool = mp.Pool(16)
    res=[]
    for f in fs:
        fname=f.split('/')[-1]
        ofname = oprefix+'outtab_'+fname
        res.append(pool.apply_async(procdesi, (f, ofname, fig_prefix)))
    for r in res:
        r.get()
        
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mask',
                        help='mask files to be fitted, i.e. /tmp/desi*fits',
                        type=str)
    parser.add_argument('oprefix',
                        help='Output prefix for the tables',
                        type=str)
    parser.add_argument('fig_prefix', 
                        help='Prefix for the fit figures',
                        type=str)
    args = parser.parser_args()
    mask = args.mask
    oprefix = args.oprefix
    fig_prefix= args.fig_prefix
    domany(mask, oprefix, fig_prefix)
    #../../desi/dc17a2/spectra-64/1/134/spectra-64-134.fits


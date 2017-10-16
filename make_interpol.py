import atpy
import numpy as np
import scipy.optimize
import numpy
import multiprocessing as mp
import matplotlib.pyplot as plt
import os
import subprocess
from idlsave import idlsave
import sys
import read_grid
import argparse

def get_revision():
    try:
        fname = os.path.dirname(os.path.realpath(__file__))
        tmpout = subprocess.Popen(
            'cd ' + fname + ' ; git log -n 1 --pretty=format:%H -- make_nd.py',
            shell=True, bufsize=80, stdout=subprocess.PIPE).stdout
        revision = tmpout.next()
    except:
        revision = ''
    return revision


git_rev = get_revision()


def get_cont(lam, spec):
    # remove the continuum trend from the template spectrum
    # by connecting  the median point of the left half of the spectrum
    # with the right half
    npix = len(lam)
    npix2 = npix // 2
    lam1, lam2 = [np.median(_) for _ in [lam[:npix2], lam[npix2:]]]
    sp1, sp2 = [np.median(_) for _ in [spec[:npix2], spec[npix2:]]]
    cont = np.exp(scipy.interpolate.UnivariateSpline([lam1, lam2],
                                                     np.log(np.r_[sp1, sp2]),
                                                     s=0, k=1, ext=0)(lam))
    return cont


class si:
    mat = None
    lamgrid = None


def processer(g, t, m, al):
    # process a spectrum at given logg temperaturem metallicity and alpha
    
    lam, spec = read_grid.get_spec(g, t, m, al)
    spec = read_grid.apply_rebinner(si.mat, spec)
    spec1 = spec / get_cont(si.lamgrid, spec)
    spec1 = np.log(spec1)  # log the spectrum
    if not numpy.isfinite(spec1).all():
        raise Exception('nans %s' % str((t, g, m, al)))
    spec1 = spec1.astype(np.float32)
    return spec1


def doit(setupInfo, postf='', dbfile='/tmp/files.db', oprefix='psavs/'):
    nthreads = 8 
    tab = atpy.Table('sqlite', dbfile)
    ids = (tab.id).astype(int)
    vec = np.array((tab.teff, tab.logg, tab.met, tab.alpha))

    i = 0

    templ_lam, spec = read_grid.get_spec(4.5, 12000, 0, 0)
    HR, lamleft, lamright, resol, step = setupInfo
    deltav = 1000.
    fac1 = (1 + deltav / (scipy.constants.speed_of_light/1e3))

    lamgrid = np.arange(lamleft / fac1, (lamright + step) * fac1, step)
    mat = read_grid.make_rebinner_new(templ_lam, lamgrid, resol)

    specs = []
    si.mat = mat
    si.lamgrid = lamgrid
    pool = mp.Pool(nthreads)
    for t, g, m, al in vec.T:
        curid = ids[i]
        i += 1
        print(i)
        specs.append(pool.apply_async(processer, (g, t, m, al)))
    lam = lamgrid
    for i in range(len(specs)):
        specs[i] = specs[i].get()

    specs = np.array(specs)
    with open('%s/specs_%s%s.pkl'%(oprefix, HR, postf),'wb') as fp:
        pickle.dump(fp, (specs, vec, lam))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument ('--setup', type=str)
    parser.add_argument('--lambda0', type=float)
    parser.add_argument('--lambda1', type=float)
    parser.add_argument('--resol',type=float)
    parser.add_argument('--step', type=float)
    parser.add_argument('--templdb',type=str,default='files.db')
    parser.add_argument('--oprefix',type=str,default='templ_data/')

    args = parser.parse_args()

    doit((args.setup, args.lambda0,args.lambda1,
        args.resol, args.step))

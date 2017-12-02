import numpy as np
import numpy.random
import scipy.interpolate
import os
import subprocess
from random import random, seed
import pickle
import dill
import argparse

def get_revision():
    """ get the git revision of the code"""
    try:
        fname = os.path.dirname(os.path.realpath(__file__))
        tmpout = subprocess.Popen(
            'cd ' + fname + ' ; git log -n 1 --pretty=format:%H -- make_nd.py',
            shell=True, bufsize=80, stdout=subprocess.PIPE).stdout
        revision = tmpout.read()
        return revision
    except:
        return ''

git_rev = get_revision()

def mapper(v):
    # map atmospheric parameters into parameters used in the grid
    import numpy as np
    return np.array([np.log10(v[0]), v[1], v[2], v[3]])

def invmapper(v):
    # inverse map
    # map grid parameters into atmospheric parameters
    import numpy as np
    return np.array([10**v[0], v[1], v[2], v[3]])

def getedgevertices(vec):
    # given a set of n-dimentional points
    # return vertices of an n-dimensional cube that fully encompass the data

    ndim = len(vec[:,0])
    span = vec.ptp(axis=1)
    lspans = vec.min(axis=1) - 0.2*span
    rspans = vec.max(axis=1) + 0.2*span
    # edges of the data
    positions = []
    # the number of vertices in the cube is 2**ndim
    for i in range(2**ndim):
        curpos = []
        for j in range(ndim):
            flag = (i & (2**j)) > 0
            curspan = [lspans[j], rspans[j]]
            curpos.append(curspan[flag])
        positions.append(curpos)
    positions = np.array(positions).T
    return positions


def dosetup(HR, prefix=None):
    "Prepare the N-d interpolation objects "
    perturb = True

    postf = ''
    with open('%s/specs_%s%s.pkl' % (prefix, HR, postf),'rb') as fp:
        D = pickle.load(fp)
        vec, specs, lam, parnames = D['vec'], D['specs'], D['lam'], D['parnames']
        del D

    vec = vec.astype(float)
    vec = mapper(vec)
    ndim = len(vec[:,0])

    if perturb:
        seed(1)
        # It turn's out that Delaunay is sometimes unstable whe
        perturbation = 1e-6
        vec = vec + np.random.uniform(-perturbation, perturbation, size=vec.shape)


    vec0 = vec.copy()

    # get the positions that are outside the existing grid
    edgepositions = getedgevertices(vec)
    vec = np.hstack((vec, edgepositions))

    nspec, lenspec = specs.shape
    fakespec = np.ones(lenspec)
    # add constant spectra to the grid at the edge locations
    specs = np.append(specs, np.tile(fakespec, (2**ndim, 1)), axis=0)

    # extra flags that allow us to detect out of the grid cases (i.e inside
    # our grid the flag should be 0)
    extraflags = np.concatenate((np.zeros(nspec), np.ones(2**ndim)))

    vec = vec.astype(np.float64)
    extraflags = extraflags.astype(np.float64)
    specs = specs.astype(np.float64)
    extraflags = extraflags[:, None]

    triang = scipy.spatial.Delaunay(vec.T)

    savefile = '%s/interp_%s%s.pkl' % (prefix, HR, postf)
    dHash = {}
    dHash['lam'] = lam
    dHash['triang'] = triang
    dHash['extraflags'] = extraflags
    dHash['vec'] = vec
    dHash['parnames'] = parnames
    dHash['mapper'] = dill.dumps(mapper)
    dHash['invmapper'] = dill.dumps(invmapper)

    with open(savefile, 'wb') as fp:
        pickle.dump(dHash, fp)
    np.save('%s/interpdat_%s%s.npy' % (prefix, HR, postf), np.asfortranarray(specs))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix',type=str)
    parser.add_argument('--setup',type=str)
    args = parser.parse_args()
    dosetup(args.setup, args.prefix)

import pickle
import argparse
import numpy as np
import numpy.random
import scipy.spatial

from rvspecfit import utils

git_rev = utils.get_revision()


def getedgevertices(vec):
    """
    Given a set of n-dimentional points, return vertices of an n-dimensional
    cube that fully encompass/surrounds the data, This is sort of the envelope
    around the data

    Parameters:
    -----------
    vec: numpy (Ndim, Npts)
        The array of input points

    Returns:
    --------
    vec: numpy (Ndim, Nretpts)
        The returned array of surrounding points
    """

    pad = 0.2 # pad each dimension by this amount (relative to the dimension width)
    ndim = len(vec[:, 0])
    span = vec.ptp(axis=1)
    lspans = vec.min(axis=1) - pad * span
    rspans = vec.max(axis=1) + pad * span
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


def execute(spec_setup, prefix=None, perturb=True):
    """
    Prepare the triangulation objects for the set of spectral data for a given
    spec_setup.

    Parameters:
    -----------
    spec_setup: string
        The spectroscopic configuration
    prefix: string
        The prefix where the data are located and where the triangulation will
        be stored
    perturb: boolean
        Boolean flag whether to perturb a little bit the points before doing a
        triangulation. This prevents issues with degenerate vertices and stability
        of triangulation. Without perturbation find_simplex for example may revert
        to brute force search.

    Returns:
    --------
    None
    """
    perturbation_amplitude = 1e-6

    postf = ''
    with open('%s/specs_%s%s.pkl' % (prefix, spec_setup, postf), 'rb') as fp:
        D = pickle.load(fp)
        vec, specs, lam, parnames, mapper = D['vec'], D['specs'], D['lam'], D['parnames'], D['mapper']
        del D

    vec = vec.astype(float)
    vec = mapper.forward(vec)
    ndim = len(vec[:, 0])

    # It turn's out that Delaunay is sometimes unstable when dealing with uniform
    # grids, so perturb points
    if perturb:
        state = np.random.get_state()
        # Make it deterministic
        np.random.seed(1)
        vec = vec + np.random.uniform(-perturbation_amplitude,
                                      perturbation_amplitude, size=vec.shape)
        np.random.set_state(state)

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

    savefile = '%s/interp_%s%s.pkl' % (prefix, spec_setup, postf)
    ret_dict = {}
    ret_dict['lam'] = lam
    ret_dict['triang'] = triang
    ret_dict['extraflags'] = extraflags
    ret_dict['vec'] = vec
    ret_dict['parnames'] = parnames
    ret_dict['mapper'] = mapper

    with open(savefile, 'wb') as fp:
        pickle.dump(ret_dict, fp)
    np.save('%s/interpdat_%s%s.npy' %
            (prefix, spec_setup, postf), np.asfortranarray(specs))


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--setup', type=str)
    args = parser.parse_args(args)
    execute(args.setup, args.prefix)

if __name__ == '__main__':
    main(sys.argv[1:])
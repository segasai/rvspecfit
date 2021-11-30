import pickle
import argparse
import sys
import numpy as np
import numpy.random
import scipy.spatial

from rvspecfit import make_interpol
import rvspecfit

git_rev = rvspecfit.__version__

INTERPOL_PKL_NAME = 'interp_%s.pkl'
INTERPOL_DAT_NAME = 'interpdat_%s.npy'


def getedgevertices(vec):
    """
    Given a set of n-dimentional points, return vertices of an n-dimensional
    cube that fully encompass/surrounds the data, This is sort of the envelope
    around the data

    Parameters
    -----------
    vec: numpy (Ndim, Npts)
        The array of input points

    Returns
    --------
    vec: numpy (Ndim, Nretpts)
        The returned array of surrounding points
    """

    pad = 0.2  # pad each dimension by this amount
    # (relative to the dimension width)
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


def execute(spec_setup, prefix=None, regular=False, perturb=True, revision=''):
    """
    Prepare the triangulation objects for the set of spectral data for a given
    spec_setup.

    Parameters
    -----------
    spec_setup: string
        The spectroscopic configuration
    prefix: string
        The prefix where the data are located and where the triangulation will
        be stored
    perturb: boolean
        Boolean flag whether to perturb a little bit the points before doing a
        triangulation. This prevents issues with degenerate vertices and
        stability of triangulation. Without perturbation find_simplex for
        example may revert to brute force search.

    Returns
    -------
    None

    """

    with open(('%s/' + make_interpol.SPEC_PKL_NAME) % (prefix, spec_setup),
              'rb') as fp:
        D = pickle.load(fp)
        (vec, specs, lam, parnames, mapper, lognorms,
         logstep) = (D['vec'], D['specs'], D['lam'], D['parnames'],
                     D['mapper'], D['lognorms'], D['logstep'])
        del D

    vec = vec.astype(float)
    vec = mapper.forward(vec)
    ndim = len(vec[:, 0])
    ret_dict = {}

    if not regular:
        perturbation_amplitude = 1e-6
        # It turn's out that Delaunay is sometimes unstable when dealing with
        # regular grids, so perturb points
        if perturb:
            state = np.random.get_state()
            # Make it deterministic
            np.random.seed(1)
            vec = vec + np.random.uniform(-perturbation_amplitude,
                                          perturbation_amplitude,
                                          size=vec.shape)
            np.random.set_state(state)

        # get the positions that are outside the existing grid
        edgepositions = getedgevertices(vec)
        nedgepos = len(edgepositions.T)

        # find out the nearest neighbors for those outside points
        nearnei = scipy.spatial.cKDTree(vec.T).query(edgepositions.T)[1]
        vec = np.hstack((vec, edgepositions))

        nspec, lenspec = specs.shape

        # add nearest neighbor sectra to the grid at the edge locations
        specs = np.append(specs, np.array([specs[_] for _ in nearnei]), axis=0)

        # extra flags that allow us to detect out of the grid cases (i.e inside
        # our grid the flag should be 0)
        extraflags = np.concatenate((np.zeros(nspec), np.ones(nedgepos)))

        lognorms = np.concatenate((lognorms, np.zeros(nedgepos)))
        vec = vec.astype(np.float64)
        extraflags = extraflags.astype(np.float64)
        specs = specs.astype(np.float64)
        extraflags = extraflags[:, None]

        triang = scipy.spatial.Delaunay(vec.T)
        ret_dict['triang'] = triang
        ret_dict['extraflags'] = extraflags

    else:
        uvecs0 = [
            np.unique(vec[i, :], return_inverse=True) for i in range(ndim)
        ]
        # Unique grid positions for each dimension
        uvecs = [_[0] for _ in uvecs0]
        # location inside uvec
        vecids = [_[1] for _ in uvecs0]
        # lens is the list of sizes [10,20, 30] with the length
        # being the number of dims
        lens = [len(_) for _ in uvecs]
        # these will be the locations in the input arrays on the grid
        # ie if we provide 3 points for [0,0], [0,1], [1,1]
        # the idgrid will be [[0,1],[-1,2]]
        idgrid = np.zeros(lens, dtype=int) - 1
        idgrid[tuple(vecids)] = np.arange(vec.shape[1])
        ret_dict['uvecs'] = uvecs
        ret_dict['regular'] = True
        ret_dict['idgrid'] = idgrid

    savefile = ('%s/' + INTERPOL_PKL_NAME) % (prefix, spec_setup)
    ret_dict['lam'] = lam
    ret_dict['logstep'] = logstep
    ret_dict['vec'] = vec
    ret_dict['parnames'] = parnames
    ret_dict['mapper'] = mapper
    ret_dict['revision'] = revision
    ret_dict['lognorms'] = lognorms
    ret_dict['git_rev'] = git_rev

    with open(savefile, 'wb') as fp:
        pickle.dump(ret_dict, fp)
    np.save(('%s/' + INTERPOL_DAT_NAME) % (prefix, spec_setup),
            np.ascontiguousarray(specs))


def main(args):
    parser = argparse.ArgumentParser(
        description='Create N-D spectral interpolation files')
    parser.add_argument(
        '--prefix',
        type=str,
        help='Location of the interpolated and convolved input spectra',
        required=True)
    parser.add_argument('--setup',
                        type=str,
                        help='Name of the spectral configuration',
                        required=True)
    parser.add_argument('--regulargrid',
                        action='store_true',
                        help='Use regular grid interpolation ',
                        required=False)
    parser.add_argument('--revision',
                        type=str,
                        help='Revision of the data files/run',
                        required=False)

    args = parser.parse_args(args)
    execute(args.setup,
            prefix=args.prefix,
            revision=args.revision,
            regular=args.regulargrid)


if __name__ == '__main__':
    main(sys.argv[1:])

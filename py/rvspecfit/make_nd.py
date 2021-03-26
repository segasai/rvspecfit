import pickle
import argparse
import sys
import ast
import numpy as np
import numpy.random
import scipy.spatial
import astropy.io.fits as pyfits
from rvspecfit import read_grid
from rvspecfit import make_interpol
import rvspecfit

git_rev = rvspecfit.__version__

INTERPOL_FITS_NAME = 'interp_%s.fits'
INTERPOL_DAT_NAME = 'interpdat_%s.npy'


class InterpParentInfo:
    def __init__(self):
        pass


class TriangInfo(InterpParentInfo):
    def __init__(
        self,
        vec=None,
        extraflags=None,
    ):
        self.triang = True
        self.regular = False
        self.vec = vec
        self.extraflags = extraflags

    def getHDUs(self):
        return [
            pyfits.ImageHDU(self.vec, name='VEC'),
            pyfits.ImageHDU(self.extraflags, name='EXTRAFLAGS')
        ]


class GridInfo(InterpParentInfo):
    def __init__(
        self,
        uvecs=None,
        idgrid=None,
    ):
        self.regular = True
        self.triang = False
        self.uvecs = uvecs
        self.idgrid = idgrid

    def getHDUs(self):
        return [
            pyfits.ImageHDU(self.uvecs, name='UVECS'),
            pyfits.ImageHDU(self.idgrid, name='IDGRID')
        ]


class InterpolInfo:
    FORMAT_VERSION = 1

    def __init__(self,
                 lam=None,
                 vec=None,
                 lognorms=None,
                 parnames=None,
                 mapper=None,
                 revision=None,
                 git_rev=None,
                 detailInfo=None,
                 cmd=None):
        self.lam = lam
        self.vec = vec
        self.lognorms = lognorms
        self.mapper = mapper
        self.parnames = parnames
        self.revision = revision
        self.git_rev = git_rev
        self.detailInfo = detailInfo
        self.cmd = cmd

    def save(self, fname):
        hdr = pyfits.Header()
        hdr['REVISION'] = self.revision
        hdr['GIT_REV'] = self.git_rev
        hdr['PARNAMES'] = str(self.parnames)
        hdr['MAPPER_LOGS'] = str(self.mapper.logs)
        hdr['MAPPER_NPARAMS'] = str(self.mapper.nparams)
        # file format
        hdr['FORMVER'] = InterpolInfo.FORMAT_VERSION
        # command used to create the file
        hdr['CMD'] = self.cmd

        lamHDU = pyfits.ImageHDU(self.lam, name='LAM')
        vecHDU = pyfits.ImageHDU(self.vec, name='VEC')
        lognormsHDU = pyfits.ImageHDU(self.lognorms, name='LOGNORMS')
        hdr['REGULAR'] = self.detailInfo.regular
        hdr['TRIANG'] = self.detailInfo.triang
        hdus = self.detailInfo.getHDUs()
        pyfits.HDUList(
            [pyfits.PrimaryHDU(header=hdr), lamHDU, vecHDU, lognormsHDU] +
            hdus).writeto(fname, overwrite=True)

    @staticmethod
    def load(fname):
        lam = pyfits.getdata(fname, 'LAM')
        vec = pyfits.getdata(fname, 'VEC')
        lognorms = pyfits.getdata(fname, 'LOGNORMS')
        hdr = pyfits.getheader(fname)
        if hdr.get('FORMVER') != InterpolInfo.FORMAT_VERSION:
            raise RuntimeError(
                '''The interpolation info file is in incompatible format.
You may need to regenerate it''')
        revision = hdr['REVISION']
        git_rev = hdr['GIT_REV']
        parnames = ast.literal_eval(hdr['PARNAMES'])
        mapper_nparams = int(hdr['MAPPER_NPARAMS'])
        mapper_logs = ast.literal_eval(hdr['MAPPER_LOGS'])
        mapper = read_grid.ParamMapper(nparams=mapper_nparams,
                                       logs=mapper_logs)
        if bool(hdr['TRIANG']):
            detailInfo = TriangInfo(
                extraflags=pyfits.getdata(fname, 'extraflags'))
        elif bool(hdr['REGULAR']):
            detailInfo = GridInfo(uvecs=pyfits.getdata(fname, 'uvecs'),
                                  idgrid=pyfits.getdata(fname, 'idgrid'))
        ii = InterpolInfo(lam=lam,
                          vec=vec,
                          lognorms=lognorms,
                          parnames=parnames,
                          mapper=mapper,
                          revision=revision,
                          git_rev=git_rev,
                          detailInfo=detailInfo)
        return ii


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


def execute(spec_setup,
            prefix=None,
            regular=False,
            perturb=True,
            revision='',
            cmd=None):
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
    cmd: string
        Command line arguments used in the call

    Returns
    -------
    None

    """

    ss = make_interpol.SpecsStore.load(
        ('%s/' + make_interpol.SPEC_FITS_NAME) % (prefix, spec_setup))
    vec, specs, lam, parnames, mapper, lognorms = (ss.vec, ss.specs, ss.lam,
                                                   ss.parnames, ss.mapper,
                                                   ss.lognorms)

    vec = vec.astype(float)
    vec = mapper.forward(vec)
    ndim = len(vec[:, 0])

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

        detailsInterp = TriangInfo(vec=vec, extraflags=extraflags)

    else:
        uvecs = [
            np.unique(vec[i, :], return_inverse=True) for i in range(ndim)
        ]
        vecids = [_[1] for _ in uvecs]
        uvecs = [_[0] for _ in uvecs]
        lens = [len(_) for _ in uvecs]
        idgrid = np.zeros(lens, dtype=int) - 1
        idgrid[tuple(vecids)] = np.arange(vec.shape[1])
        detailsInterp = GridInfo(uvecs=uvecs, idgrid=idgrid)

    savefile = ('%s/' + INTERPOL_FITS_NAME) % (prefix, spec_setup)
    ii = InterpolInfo(lam=lam,
                      vec=vec,
                      lognorms=lognorms,
                      parnames=parnames,
                      revision=revision,
                      git_rev=git_rev,
                      mapper=mapper,
                      detailInfo=detailsInterp,
                      cmd=cmd)
    ii.save(savefile)
    del ii

    np.save(('%s/' + INTERPOL_DAT_NAME) % (prefix, spec_setup),
            np.ascontiguousarray(specs))


def main(args):
    cmd = ' '.join(args)
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
            regular=args.regulargrid,
            cmd=cmd)


if __name__ == '__main__':
    main(sys.argv[1:])

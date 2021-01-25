import numpy as np
import scipy.spatial
import scipy.interpolate
import pickle
import itertools
from rvspecfit import make_nd


class TriInterp:
    def __init__(self, triang, dats, exp=True):
        """
        Get the Interpolation object from the Delaunay triangulation
        and array of vectors

        Parameters
        ----------

        triang: Delaunay triangulation
            Triangulation from scipy.spatial
        dats: ndarray
            2d array of vectors to be interpolated
        exp: bool
            if True the output needs to be exponentiated

        """
        self.triang = triang
        self.dats = dats
        self.exp = exp
        self.ndim = self.triang.ndim
        self.b1 = np.empty(self.ndim + 1, dtype=self.dats.dtype)

    def __call__(self, p):
        """ Compute the interpolated spectrum at parameter vector p

        Parameters
        ----------
        p: ndarray
            1-D numpy array of parameters

        """
        p = np.asarray(p)
        xid = self.triang.find_simplex(p)
        if xid == -1:
            return np.nan
        b1 = self.b1
        ndim = self.ndim
        b1[:ndim] = self.triang.transform[xid, :ndim, :].dot(
            p - self.triang.transform[xid, ndim, :])
        b1[ndim] = 1 - b1[:ndim].sum()
        spec = (self.dats[self.triang.simplices[xid], :] *
                b1[:, None]).sum(axis=0)
        if self.exp:
            spec = np.exp(spec)
        return spec


class GridOutsideCheck:
    def __init__(self, uvecs, vecs, idgrid):
        self.uvecs = uvecs
        self.idgrid = idgrid
        self.ndim = len(self.uvecs)
        edges = itertools.product(*[[0, 1] for i in range(self.ndim)])
        self.edges = [np.array(_) for _ in edges]  # 0,1,0,1 vectors
        self.lens = np.array([len(_) for _ in self.uvecs])
        self.Ns = self.idgrid.shape
        self.ptp = vecs.ptp(axis=1)
        self.tree = scipy.spatial.cKDTree(vecs.T / self.ptp[None, :],
                                          compact_nodes=False,
                                          balanced_tree=False)

    def __call__(self, p):
        # ATM that doesn't return something that tells us how far away
        # we are
        pos = np.array([
            np.searchsorted(self.uvecs[i], p[i], 'right') - 1
            for i in range(self.ndim)
        ])
        outside = False
        if np.any((pos < 0) | (pos >= self.lens - 1)):
            outside = True
        else:
            if (self.idgrid[tuple((pos[None, :] + self.edges).T)] == -1).any():
                outside = True
        if outside:
            return self.tree.query(p / self.ptp)[0]
        return 0


class GridInterp:
    def __init__(self, uvecs, idgrid, vecs, dats, exp=True):
        """
        Get the Grid interpolation object

        Parameters
        ----------

        uvecs: List of unique grid values
        idgrid: grid of spectral ids (-1 if not avalaible
        dats: ndarray
            2d array of vectors to be interpolated
        exp: bool
            if True the output needs to be exponentiated

        """
        self.uvecs = uvecs
        self.dats = dats
        self.exp = exp
        self.idgrid = idgrid
        self.ndim = len(self.uvecs)
        self.lens = np.array([len(_) for _ in self.uvecs])
        edges = itertools.product(*[[0, 1] for i in range(self.ndim)])
        self.edges = np.array([np.array(_) for _ in edges])  # 0,1,0,1 vectors

        self.ptp = vecs.ptp(axis=1)
        self.tree = scipy.spatial.cKDTree(vecs.T / self.ptp[None, :])

    def get_nearest(self, p):
        return self.tree.query(p / self.ptp)[1]

    def __call__(self, p):
        """ Compute the interpolated spectrum at parameter vector p

        Parameters
        ----------
        p: ndarray
            1-D numpy array of parameters

        """
        p = np.asarray(p)
        ndim = self.ndim

        # wrapper to exponentiate when needed
        if self.exp:
            FF = np.exp
        else:
            FF = lambda x: x

        # gridlocs
        pos = np.array(
            [np.digitize(p[i], self.uvecs[i]) - 1 for i in range(ndim)])
        if np.any((pos < 0) | (pos >= (self.lens - 1))):
            ret = self.get_nearest(p)
            return FF(self.dats[ret])

        coeffs = np.array([(p[i] - self.uvecs[i][pos[i]]) /
                           (self.uvecs[i][pos[i] + 1] - self.uvecs[i][pos[i]])
                           for i in range(ndim)])  # from 0 to 1
        # ndim  vec
        coeffs2 = np.zeros((2**ndim, self.ndim))
        pos2 = np.zeros(2**ndim, dtype=int)
        coeffs2 = coeffs[None, :]**self.edges * (1 - coeffs[None, :])**(
            1 - self.edges)
        pos2 = self.idgrid[tuple((pos[None, :] + self.edges).T)]
        coeffs2 = np.prod(coeffs2, axis=1)
        if np.any(pos2 < 0):
            # outside boundary
            ret = self.get_nearest(p)
            return FF(self.dats[ret])
        spec = (np.dot(coeffs2, self.dats[pos2, :]))
        return FF(spec)


class SpecInterpolator:
    """ Spectrum interpolator object """
    def __init__(self,
                 name,
                 interper,
                 extraper,
                 lam,
                 mapper,
                 parnames,
                 revision='',
                 filename='',
                 creation_soft_version=''):
        """ Construct the interpolator object

        Parameters
        ----------
        name: string
            The name of spectroscopic configuration (instrument arm)
        interper: function
            The interpolating function from parameters to spectrum
        extraper: function
            Function that returns non-zero if outside the box
        lam: ndarray
            Wavelength vector
        mapper: function
            Function that does the mapping from scaled box parameters to 
            proper values
        parnames: tuple
            The list of parameter names ('logg', 'teff' ,.. ) etc
        revision: str
            The revision of the grid
        filename: str
            The filename from which the template was read
        creation_soft_version: str
            The version of soft used to create the interpolator

        """

        self.name = name
        self.lam = lam
        self.interper = interper
        self.extraper = extraper
        self.mapper = mapper
        self.parnames = parnames
        self.revision = revision
        self.filename = filename
        self.creation_soft_version = creation_soft_version

    def outsideFlag(self, param0):
        """Check if the point is outside the interpolation grid

        Parameters
        ----------
        param0: tuple
            parameter vector

        Returns
        ret: float
            > 0 if point outside the grid

        """
        param = self.mapper.forward(param0)
        return self.extraper(param)

    def eval(self, param0):
        """ Evaluate the spectrum at a given parameter """
        if isinstance(param0, dict):
            param0 = [param0[_] for _ in self.parnames]
        param = self.mapper.forward(param0)
        return self.interper(param)


class interp_cache:
    """ Singleton object caching the interpolators
    """
    interps = {}


def getInterpolator(HR, config, warmup_cache=True):
    """ Return the spectrum interpolation object for a given instrument
    setup HR and config. This function also checks the cache

    Parameters
    ----------
    HR: string
        Spectral configuration
    config: dict
        Configuration
    warmup_cache: bool
        If True we read the whole file to warm up the OS cache

    Returns
    -------
    ret: SpecInterpolator
        The spectral interpolator

    """
    if HR not in interp_cache.interps:
        expFlag = True
        dats = np.load(config['template_lib'] + make_nd.INTERPOL_DAT_NAME % HR,
                       mmap_mode='r')
        if warmup_cache:
            # we read all the templates to put them in the memory cache
            dats.sum()

        savefile = config['template_lib'] + make_nd.INTERPOL_PKL_NAME % HR
        with open(savefile, 'rb') as fd0:
            fd = pickle.load(fd0)
        (templ_lam, vecs, mapper, parnames) = (fd['lam'], fd['vec'],
                                               fd['mapper'], fd['parnames'])

        if 'triang' in fd:
            # triangulation based interpolation
            (triang, extraflags) = (fd['triang'], fd['extraflags'])
            interper, extraper = (TriInterp(triang, dats, exp=expFlag),
                                  TriInterp(triang, extraflags, exp=False))
        elif 'regular' in fd:
            # regular grid interpolation
            uvecs, idgrid = (fd['uvecs'], fd['idgrid'])
            interper = GridInterp(uvecs, idgrid, vecs, dats, exp=expFlag)
            extraper = GridOutsideCheck(uvecs, vecs, idgrid)
        else:
            raise RuntimeError('Unrecognized interpolation file')
        revision = fd.get('revision') or ''
        creation_soft_version = fd.get('git_rev') or ''
        interpObj = SpecInterpolator(
            HR,
            interper,
            extraper,
            templ_lam,
            mapper,
            parnames,
            revision=revision,
            creation_soft_version=creation_soft_version,
            filename=savefile)
        interp_cache.interps[HR] = interpObj
    else:
        interpObj = interp_cache.interps[HR]
    return interpObj


def getSpecParams(setup, config):
    ''' Return the ordered list of spectral parameters
    of a given spectroscopic setup

    Parameters
    ----------
    setup: str
        Spectral configuration
    config: dict
        Configuration dictionary

    Returns
    -------
    ret: list
        List of parameters names
    '''
    return getInterpolator(setup, config).parnames

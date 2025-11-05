import numpy as np
import scipy.spatial
import scipy.interpolate
import itertools
import importlib
from rvspecfit import make_nd
from rvspecfit import make_interpol
from rvspecfit import serializer


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
        if spec.size == 1:
            spec = float(spec[0])
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
        self.ptp = np.ptp(vecs, axis=1)
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

        uvecs: List of unique grid values for each dim
        idgrid: grid of spectral ids
        This should be the full ndim-dimensional grid of ints.
        I.e it'll be the id of spectrum from dats, or -1 if the
        spectrum at that gridpoint is not known
        vecs: ndarray
        these are original coordinates of each spectrum (from dats)
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
        self.edges = np.array([np.array(_) for _ in edges])
        # list of vectors corresponding to vertices of unit cube, i.e.
        # [[0,0], [0,1], [1,0], [1, 1]]

        self.ptp = np.ptp(vecs, axis=1)
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

        # these are integer position in each dimension
        pos = np.array(
            [np.digitize(p[i], self.uvecs[i]) - 1 for i in range(ndim)])
        if np.any((pos < 0) | (pos >= (self.lens - 1))):
            if not np.isfinite(p).all():
                # this can happen if teff<0
                # and p is not finite i just take the first spectrum
                ret = 0
            else:
                ret = self.get_nearest(p)
            return FF(self.dats[ret])

        # here we check that all the spectra at the vertices
        # are known
        pos2 = self.idgrid[tuple((pos[None, :] + self.edges).T)]
        if np.any(pos2 < 0):
            # outside boundary
            ret = self.get_nearest(p)
            return FF(self.dats[ret])

        # The logic here is following.
        # this is 2d polylinear interpolation
        # V00 * ( 1-x) *( 1-y) + V01 * (1-x) * y + V10* x*(1-y) + V11 *x*y
        # I.e. V00 * x^0 * y^0 * (1-x)^1 * (1-y) + ...
        # if we have unit n-d cube with vertices at bitstrings S
        # then the interpolation is V_S * X^S * (1-X)^(1-S)
        # where X^S is the vector power i.e. Product_i(X[i]^S[i])

        coeffs = np.array([(p[i] - self.uvecs[i][pos[i]]) /
                           (self.uvecs[i][pos[i] + 1] - self.uvecs[i][pos[i]])
                           for i in range(ndim)])  # from 0 to 1
        # these are essentially normalized x_i values

        # This is the array of X_i^S_i * (1-X_i)^S_i
        # the first dimension correspond to different vertices (therefore
        # different S strings) and second dimension correspond to dimension
        # 1...ndim we then need to take product over the last dim
        coeffs2 = coeffs[None, :]**self.edges * (1 - coeffs[None, :])**(
            1 - self.edges)
        coeffs2 = np.prod(coeffs2, axis=1)
        # the final sum of coefficients times spectra at cube vertices
        spec = np.dot(coeffs2, self.dats[pos2, :])
        return FF(spec)


class SpecInterpolator:
    """ Spectrum interpolator object

    .. autosummary::
        eval
        outsideFlag
    """

    def __init__(self,
                 name,
                 interper,
                 extraper,
                 lam,
                 mapper,
                 parnames,
                 revision='',
                 filename='',
                 creation_soft_version='',
                 log_step=None):
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
            Function that does the mapping from external parameters to
            internal represetnation, i.e. going from teff,logg to
            logteff, logg
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
        self.parnames = tuple(parnames)
        self.revision = revision
        self.filename = filename
        self.creation_soft_version = creation_soft_version
        self.log_step = log_step
        self.objid = hash((self.name, self.parnames, self.revision,
                           self.filename, self.creation_soft_version))

    def __hash__(self):
        return self.objid

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
            try:
                param0 = [param0[_] for _ in self.parnames]
            except KeyError as exc:
                missing_key = exc.args[0]
                raise ValueError(f'The parameter {missing_key} not found. '
                                 "Required list of parameters is: " +
                                 (','.join(self.parnames)))
        param = self.mapper.forward(param0)
        return self.interper(param)


class interp_cache:
    """ Singleton object caching the interpolators
    """
    interps = {}
    template_lib = None


def getInterpolator(HR, config, warmup_cache=False, cache=None):
    """ Return the spectrum interpolation object for a given instrument
    setup HR and config. This function also checks the cache

    Parameters
    ----------
    HR: string
        Spectral configuration
    config: dict
        Configuration
    cache: dict or None
        Dictionary like object with the cache. If None, internal cache is
        used instead.
    warmup_cache: bool
        If True we read the whole file to warm up the OS cache

    Returns
    -------
    ret: SpecInterpolator
        The spectral interpolator

    """
    if cache is None:
        cache = interp_cache.interps
        system_cache = True
        if config['template_lib'] != interp_cache.template_lib:
            # clear old cache
            interp_cache.template_lib = config['template_lib']
            interp_cache.interps = {}
    else:
        system_cache = False
    if HR not in cache:
        savefile = (config['template_lib'] + '/' +
                    make_nd.INTERPOL_H5_NAME % HR)
        fd = serializer.load_dict_from_hdf5(savefile)
        log_spec = fd.get('log_spec') or True

        (templ_lam, parnames) = (fd['lam'], fd['parnames'])
        mapper_module = fd['mapper_module']
        mapper_class_name = fd['mapper_class_name']
        mapper_args = fd['mapper_args']
        mapper = make_interpol.get_mapper(mapper_module, mapper_class_name,
                                          mapper_args)

        log_step = fd['log_step']

        if 'interpolation_type' in fd:
            interp_type = fd['interpolation_type']
        else:
            if 'triang' in fd:
                interp_type = 'triangulation'
            elif 'regular' in fd:
                interp_type = 'regulargrid'
            else:
                raise RuntimeError('Unrecognized interpolation file')

        if interp_type in ['triangulation', 'regulargrid']:
            dats = np.load(config['template_lib'] + '/' +
                           make_nd.INTERPOL_DAT_NAME % HR,
                           mmap_mode='r')
            if warmup_cache:
                # we read all the templates to put them in the memory cache
                dats.sum()
            vecs = fd['vec']

        if interp_type == 'triangulation':
            # triangulation based interpolation
            (triang, extraflags) = (fd['triang'], fd['extraflags'])
            interper, extraper = (TriInterp(triang, dats, exp=log_spec),
                                  TriInterp(triang, extraflags, exp=False))
        elif interp_type == 'regulargrid':
            # regular grid interpolation
            uvecs, idgrid = (fd['uvecs'], fd['idgrid'])
            interper = GridInterp(uvecs, idgrid, vecs, dats, exp=log_spec)
            extraper = GridOutsideCheck(uvecs, vecs, idgrid)
        elif interp_type == 'generic':
            mod_name = fd['module']
            class_name = fd['class_name']
            outside_class_name = fd['outside_class_name']
            mod = importlib.import_module(mod_name)
            fd['template_lib'] = config['template_lib']
            interper = getattr(mod, class_name)(fd)
            extraper = getattr(mod, outside_class_name)(fd)
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
            filename=savefile,
            log_step=log_step)
        cache[HR] = interpObj
        if system_cache:
            interp_cache.template_lib = config['template_lib']
    return cache[HR]


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

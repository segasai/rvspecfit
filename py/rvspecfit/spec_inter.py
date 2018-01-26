import numpy as np
import scipy.spatial
import scipy.interpolate
import pickle

from rvspecfit import make_nd

def getInterp(triang, dats, exp=True):
    # get Interpolation object from the Delaunay triangulation
    # and array of vectors
    def func(p):
        p = np.asarray(p)
        ndim = triang.ndim
        xid = triang.find_simplex(p)
        if xid == -1:
            return np.nan
        b = triang.transform[xid, :ndim, :].dot(
            p - triang.transform[xid, ndim, :])
        b1 = np.r_[b, [1 - b.sum()]]
        spec = (dats[triang.simplices[xid], :] * b1[:, None]).sum(axis=0)
        if exp:
            spec = np.exp(spec)
        return spec
    return func


class SpecInterpolator:
        # Spectrum interpolator object
    def __init__(self, name, interper, extraper, lam, mapper,
                 parnames):
        """ Construct the interpolator object
        The arguments are the name of the instrument setup
        The interpolator object that returns the
        The extrapolation object,
        """

        self.name = name
        self.lam = lam
        self.interper = interper
        self.extraper = extraper
        self.mapper = mapper
        self.parnames = parnames

    def outsideFlag(self, param0):
        """Check if the point is outside the interpolation grid"""
        param = self.mapper.forward(param0)
        return self.extraper(param)

    def eval(self, param0):
        """ Evaluate the spectrum at a given parameter """
        if isinstance(param0, dict):
            param0 = [param0[_] for _ in self.parnames]
        param = self.mapper.forward(param0)
        return self.interper(param)


class interp_cache:
    interps = {}


def getInterpolator(HR, config, warmup_cache=True):
    """ return the spectrum interpolation object for a given instrument
    setup HR and config
    """
    if HR not in interp_cache.interps:
        savefile = config['template_lib'] +  make_nd.INTERPOL_PKL_NAME % HR
        with open(savefile, 'rb') as fd0:
            fd = pickle.load(fd0)
            (triang, templ_lam, vecs, extraflags, mapper, parnames) = (
                fd['triang'], fd['lam'], fd['vec'], fd['extraflags'],
                fd['mapper'], fd['parnames'])
        expFlag = True
        dats = np.load(config['template_lib']+ make_nd.INTERPOL_DAT_NAME % HR,
                       mmap_mode='r')
        if warmup_cache:
            # we read all the templates to put them in the memory cache
            dat.sum()
        interper, extraper = (getInterp(triang, dats, exp=expFlag),
                              scipy.interpolate.LinearNDInterpolator(triang, extraflags))
        interpObj = SpecInterpolator(HR, interper, extraper, templ_lam,
                                     mapper, parnames)
        interp_cache.interps[HR] = interpObj
    else:
        interpObj = interp_cache.interps[HR]
    return interpObj


def getSpecParams(setup, config):
    ''' Return the ordered list of spectral parameters
    of a given spectroscopic setup'''
    return getInterpolator(setup, config).parnames

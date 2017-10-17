import numpy as np
import scipy.spatial
import scipy.interpolate
import pickle
import dill


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
    def __init__(self, name, interper, extraper, lam, mapper, invmapper):
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
        self.invmapper = invmapper

    def outsideFlag(self, param0):
        """Check if the point is outside the interpolation grid"""
        param = self.mapper(param0)
        return self.extraper(param)

    def eval(self, param0):
        """ Evaluate the spectrum at a given parameter """
        param = self.mapper(param0)
        return self.interper(param)


class interp_cache:
    interps = {}


def getInterpolator(HR, config):
    """ return the spectrum interpolation object for a given instrument
    setup HR and config
    """ 
    if HR not in interp_cache.interps:
        with open(config['template_lib']['savefile'] % HR, 'rb') as fd0:
            fd = pickle.load(fd0)
            (triang, templ_lam, vecs, extraflags, mapper, invmapper) = (
                fd['triang'], fd['lam'], fd['vec'], fd['extraflags'],
                fd['mapper'], fd['invmapper'])
            mapper = dill.loads(mapper)
            invmapper = dill.loads(invmapper)
        expFlag = True
        dats = np.load(config['template_lib']['npyfile'] % HR,
                       mmap_mode='r')
        interper, extraper = (getInterp(triang, dats, exp=expFlag),
                              scipy.interpolate.LinearNDInterpolator(triang, extraflags))
        interpObj = SpecInterpolator(HR, interper, extraper, templ_lam,
                                     mapper, invmapper)
        interp_cache.interps[HR] = interpObj
    else:
        interpObj = interp_cache.interps[HR]
    return interpObj

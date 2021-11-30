import numpy as np

from . import _spliner

ffi = _spliner.ffi


class Spline:
    def __init__(self, xs, ys, logstep=True):
        assert (xs.flags['C_CONTIGUOUS'])
        assert (ys.flags['C_CONTIGUOUS'])
        assert (xs.dtype == np.float64)
        assert (ys.dtype == np.float64)
        N = len(xs)
        A = np.zeros(N - 1, dtype=np.float64)
        B = np.zeros(N - 1, dtype=np.float64)
        C = np.zeros(N - 1, dtype=np.float64)
        D = np.zeros(N - 1, dtype=np.float64)
        h = np.zeros(N - 1, dtype=np.float64)

        self.logstep = int(logstep)
        _spliner.lib.construct(ffi.from_buffer('double *', xs),
                               ffi.from_buffer('double *', ys), N,
                               ffi.from_buffer('double *', A),
                               ffi.from_buffer('double *', B),
                               ffi.from_buffer('double *', C),
                               ffi.from_buffer('double *', D),
                               ffi.from_buffer('double *', h))

        self.N, self.A, self.B, self.C, self.D, self.h = N, A, B, C, D, h
        self.xs = xs

    def __call__(self, evalx):
        assert (evalx.flags['C_CONTIGUOUS'])
        assert (evalx.dtype == np.float64)

        nevalx = len(evalx)
        ret = np.zeros(nevalx, dtype=np.float64)
        nevalx = len(evalx)
        stat = _spliner.lib.evaler(ffi.from_buffer('double *',
                                                   evalx), nevalx, self.N,
                                   ffi.from_buffer('double *', self.xs),
                                   ffi.from_buffer('double *', self.h),
                                   ffi.from_buffer('double *', self.A),
                                   ffi.from_buffer('double *', self.B),
                                   ffi.from_buffer('double *', self.C),
                                   ffi.from_buffer('double *', self.D),
                                   self.logstep,
                                   ffi.from_buffer('double *', ret))
        assert (stat == 0)

        return ret

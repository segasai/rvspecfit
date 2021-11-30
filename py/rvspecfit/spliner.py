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

        xs_c = ffi.cast('double *', ffi.from_buffer(xs))
        ys_c = ffi.cast('double *', ffi.from_buffer(ys))
        A_c = ffi.cast('double *', ffi.from_buffer(A))
        B_c = ffi.cast('double *', ffi.from_buffer(B))
        C_c = ffi.cast('double *', ffi.from_buffer(C))
        D_c = ffi.cast('double *', ffi.from_buffer(D))
        h_c = ffi.cast('double *', ffi.from_buffer(h))
        self.logstep = int(logstep)
        _spliner.lib.construct(xs_c, ys_c, N, A_c, B_c, C_c, D_c, h_c)

        self.N, self.A, self.B, self.C, self.D, self.h = N, A, B, C, D, h
        self.xs = xs

    def __call__(self, evalx):
        assert (evalx.flags['C_CONTIGUOUS'])
        assert (evalx.dtype == np.float64)

        nevalx = len(evalx)
        ret = np.zeros(nevalx, dtype=np.float64)
        xs_c = ffi.cast('double *', ffi.from_buffer(self.xs))
        A_c = ffi.cast('double *', ffi.from_buffer(self.A))
        B_c = ffi.cast('double *', ffi.from_buffer(self.B))
        C_c = ffi.cast('double *', ffi.from_buffer(self.C))
        D_c = ffi.cast('double *', ffi.from_buffer(self.D))
        h_c = ffi.cast('double *', ffi.from_buffer(self.h))
        evalx_c = ffi.cast('double *', ffi.from_buffer(evalx))
        ret_c = ffi.cast('double *', ffi.from_buffer(ret))
        nevalx = len(evalx)
        stat = _spliner.lib.evaler(evalx_c, nevalx, self.N, xs_c, h_c, A_c,
                                   B_c, C_c, D_c, self.logstep, ret_c)
        assert (stat == 0)

        return ret

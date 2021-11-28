"""Interpolation algorithms using piecewise cubic polynomials.

Code taken from scipy.

Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from scipy.interpolate import PPoly
from scipy.linalg import solve_banded


class CubicSpline(PPoly):
    """Cubic spline data interpolator.

    Interpolate data with a piecewise cubic polynomial which is twice
    continuously differentiable [1]_. The result is represented as a `PPoly`
    instance with breakpoints matching the given data.

    Parameters
    ----------
    x : array_like, shape (n,)
        1-D array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    y : array_like
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along ``axis``
        (see below) must match the length of ``x``. Values must be finite.
    extrapolate : {bool, 'periodic', None}, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. If None (default), ``extrapolate`` is
        set to 'periodic' for ``bc_type='periodic'`` and to True otherwise.

    Attributes
    ----------
    x : ndarray, shape (n,)
        Breakpoints. The same ``x`` which was passed to the constructor.
    c : ndarray, shape (4, n-1, ...)
        Coefficients of the polynomials on each segment. The trailing
        dimensions match the dimensions of `y`, excluding ``axis``.
        For example, if `y` is 1-d, then ``c[k, i]`` is a coefficient for
        ``(x-x[i])**(3-k)`` on the segment between ``x[i]`` and ``x[i+1]``.
    axis : int
        Interpolation axis. The same axis which was passed to the
        constructor.

    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    roots

    Notes
    References
    ----------
    .. [1] `Cubic Spline Interpolation
            <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_
            on Wikiversity.
    .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.
    """
    def __init__(self, x, y, extrapolate=None):
        dx = np.diff(x)

        n = len(x)

        # Find derivative values at each x[i] by solving a tridiagonal
        # system.
        A = np.zeros((3, n))  # This is a banded matrix representation.

        A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The diagonal
        A[0, 2:] = dx[:-1]  # The upper diagonal
        A[-1, :-2] = dx[1:]  # The lower diagonal
        A[1, 0] = dx[1]
        A[0, 1] = x[2] - x[0]
        A[1, -1] = dx[-2]
        A[-1, -2] = x[-1] - x[-3]

        slope = np.diff(y) / dx
        b = np.empty(n, dtype=y.dtype)
        b[1:-1] = 3 * (dx[1:] * slope[:-1] + dx[:-1] * slope[1:])

        d = x[2] - x[0]
        b[0] = ((dx[0] + 2 * d) * dx[1] * slope[0] + dx[0]**2 * slope[1]) / d
        d = x[-1] - x[-3]
        b[-1] = ((dx[-1]**2 * slope[-2] +
                  (2 * d + dx[-1]) * dx[-2] * slope[-1]) / d)

        s = solve_banded((1, 1),
                         A,
                         b,
                         overwrite_ab=True,
                         overwrite_b=True,
                         check_finite=False)

        dydx = s
        t = (dydx[:-1] + dydx[1:] - 2 * slope) / dx
        c = np.empty((4, len(x) - 1), dtype=t.dtype)
        c[0] = t / dx
        c[1] = (slope - dydx[:-1]) / dx - t
        c[2] = dydx[:-1]
        c[3] = y[:-1]
        super(CubicSpline, self).__init__(c, x, extrapolate=extrapolate)
        self.axis = 0

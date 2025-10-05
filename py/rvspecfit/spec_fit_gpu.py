"""
GPU-accelerated spectral fitting routines using CuPy

This module provides GPU implementations of the core fitting functions
from spec_fit.py for batch processing of spectra.
"""

import numpy as np
try:
    import cupy as cp
    import cupyx.scipy.sparse
    import cupyx.scipy.linalg
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

from rvspecfit import spec_fit
import functools


class CubicSplineGPU:
    """
    GPU-accelerated cubic spline interpolation
    Replaces the C-based spliner with CuPy implementation
    """

    def __init__(self, xs, ys, log_step=True, device_id=0):
        """
        Parameters
        ----------
        xs : array_like
            Knot positions (must be monotonically increasing)
        ys : array_like
            Values at knots
        log_step : bool
            If True, knots are uniformly spaced in log space
        device_id : int
            GPU device ID to use
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available. Install cupy for GPU support.")

        with cp.cuda.Device(device_id):
            self.device_id = device_id
            self.log_step = log_step

            # Move data to GPU
            xs_gpu = cp.asarray(xs, dtype=cp.float64)
            ys_gpu = cp.asarray(ys, dtype=cp.float64)

            N = len(xs_gpu)

            # Compute spline coefficients using Thomas algorithm
            # (tridiagonal matrix solver)
            h = xs_gpu[1:] - xs_gpu[:-1]
            hinv = 1.0 / h
            bs = (ys_gpu[1:] - ys_gpu[:-1]) * hinv

            N1 = N - 1
            N2 = N - 2
            N3 = N - 3

            # Build tridiagonal system
            vs = 2 * (h[1:] + h[:-1])  # diagonal
            us = 6 * (bs[1:] - bs[:-1])  # RHS

            # Thomas algorithm (forward elimination)
            cc = h[1:]  # off-diagonal
            bb = vs
            dd = us

            cc_dash = cp.zeros(N3, dtype=cp.float64)
            dd_dash = cp.zeros(N2, dtype=cp.float64)

            cc_dash[0] = cc[0] / bb[0]
            for i in range(1, N3):
                cc_dash[i] = cc[i] / (bb[i] - cc[i-1] * cc_dash[i-1])

            dd_dash[0] = dd[0] / bb[0]
            for i in range(1, N2):
                dd_dash[i] = (dd[i] - cc[i-1] * dd_dash[i-1]) / \
                             (bb[i] - cc[i-1] * cc_dash[i-1])

            # Back substitution
            zs = cp.zeros(N, dtype=cp.float64)
            zs[N2] = dd_dash[N3]
            for i in range(N3-1, -1, -1):
                zs[i+1] = dd_dash[i] - cc_dash[i] * zs[i+2]

            # Compute spline coefficients
            one_sixth = 1.0 / 6.0
            A = cp.zeros(N1, dtype=cp.float64)
            B = cp.zeros(N1, dtype=cp.float64)
            C = cp.zeros(N1, dtype=cp.float64)
            D = cp.zeros(N1, dtype=cp.float64)

            for i in range(N1):
                tmp1 = hinv[i] * one_sixth
                tmp2 = h[i] * one_sixth
                A[i] = zs[i+1] * tmp1
                B[i] = zs[i] * tmp1
                C[i] = ys_gpu[i+1] * hinv[i] - zs[i+1] * tmp2
                D[i] = ys_gpu[i] * hinv[i] - zs[i] * tmp2

            self.xs = xs_gpu
            self.h = h
            self.A = A
            self.B = B
            self.C = C
            self.D = D
            self.N = N

    def __call__(self, evalx):
        """
        Evaluate spline at positions evalx

        Parameters
        ----------
        evalx : array_like
            Positions to evaluate spline at

        Returns
        -------
        ret : ndarray
            Spline values at evalx
        """
        with cp.cuda.Device(self.device_id):
            evalx_gpu = cp.asarray(evalx, dtype=cp.float64)
            nevalx = len(evalx_gpu)

            # Find positions in knot array
            x0 = self.xs[0]

            if self.log_step:
                logstep = cp.log(self.xs[1] / x0)
                logx0 = cp.log(x0)
                pos = ((cp.log(evalx_gpu) - logx0) / logstep).astype(cp.int32)
            else:
                step = self.xs[1] - x0
                pos = ((evalx_gpu - x0) / step).astype(cp.int32)

            # Clip to valid range
            pos = cp.clip(pos, 0, self.N - 2)

            # Evaluate cubic spline
            dxl = evalx_gpu - self.xs[pos]
            dxr = self.xs[pos + 1] - evalx_gpu

            ret = (self.A[pos] * dxl * dxl * dxl +
                   self.B[pos] * dxr * dxr * dxr +
                   self.C[pos] * dxl +
                   self.D[pos] * dxr)

            return ret


def get_chisq0_batch_gpu(specs, templs, polys, especs=None, device_id=0):
    """
    Batched GPU version of get_chisq0 for multiple spectra

    Parameters
    ----------
    specs : array_like, shape (n_spectra, n_pixels)
        Batch of observed spectra
    templs : array_like, shape (n_spectra, n_pixels)
        Batch of template spectra
    polys : array_like, shape (n_poly, n_pixels)
        Polynomial basis functions
    especs : array_like, shape (n_spectra, n_pixels), optional
        Error spectra
    device_id : int
        GPU device ID

    Returns
    -------
    chisqs : array_like, shape (n_spectra,)
        Chi-square values
    coeffs : array_like, shape (n_spectra, n_poly)
        Best-fit polynomial coefficients
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    with cp.cuda.Device(device_id):
        specs_gpu = cp.asarray(specs, dtype=cp.float64)
        templs_gpu = cp.asarray(templs, dtype=cp.float64)
        polys_gpu = cp.asarray(polys, dtype=cp.float64)

        n_spectra, n_pixels = specs_gpu.shape
        n_poly = polys_gpu.shape[0]

        # Normalize by errors if provided
        if especs is not None:
            especs_gpu = cp.asarray(especs, dtype=cp.float64)
            normspecs = specs_gpu / especs_gpu
            normtempls = templs_gpu / especs_gpu
            logl_z = cp.sum(cp.log(especs_gpu), axis=1)
        else:
            normspecs = specs_gpu
            normtempls = templs_gpu
            logl_z = cp.zeros(n_spectra, dtype=cp.float64)

        # Batch compute polynomial fits
        # polys1[i, j, k] = templ[i, k] * poly[j, k]
        polys1 = normtempls[:, None, :] * polys_gpu[None, :, :]

        # vector1[i, j] = sum_k polys1[i, j, k] * normspec[i, k]
        vector1 = cp.sum(polys1 * normspecs[:, None, :], axis=2)

        # matrix1[i, j, k] = sum_l polys1[i, j, l] * polys1[i, k, l]
        # This is batched matrix multiply: (n_spectra, n_poly, n_pixels) @ (n_spectra, n_pixels, n_poly)
        matrix1 = cp.einsum('ijk,ilk->ijl', polys1, polys1)

        # SVD decomposition for each spectrum
        chisqs = cp.zeros(n_spectra, dtype=cp.float64)
        coeffs_batch = cp.zeros((n_spectra, n_poly), dtype=cp.float64)

        for i in range(n_spectra):
            u, s, vh = cp.linalg.svd(matrix1[i], full_matrices=False)
            ldetI = cp.sum(cp.log(s))

            # Solve using SVD
            v2 = vh.T @ (cp.diag(1.0 / s) @ (u.T @ vector1[i]))

            chisq = (-ldetI + 2 * logl_z[i] +
                     cp.linalg.norm(normspecs[i] - v2 @ polys1[i])**2)

            chisqs[i] = chisq
            coeffs_batch[i] = v2

        return chisqs, coeffs_batch


def convolve_vsini_batch_gpu(lam_templ, templs, vsinis, device_id=0):
    """
    Batch convolution of templates with rotation kernels on GPU

    Parameters
    ----------
    lam_templ : array_like
        Wavelength array (must be log-spaced)
    templs : array_like, shape (n_spectra, n_pixels)
        Template spectra
    vsinis : array_like, shape (n_spectra,)
        Rotation velocities for each spectrum
    device_id : int
        GPU device ID

    Returns
    -------
    convolved : array_like, shape (n_spectra, n_pixels)
        Convolved spectra
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    with cp.cuda.Device(device_id):
        lam_gpu = cp.asarray(lam_templ, dtype=cp.float64)
        templs_gpu = cp.asarray(templs, dtype=cp.float64)
        vsinis_gpu = cp.asarray(vsinis, dtype=cp.float64)

        SPEED_OF_LIGHT = 299792.458  # km/s
        eps = 0.6  # limb darkening

        lnstep = cp.log(lam_gpu[1] / lam_gpu[0])

        n_spectra = templs_gpu.shape[0]
        n_pixels = templs_gpu.shape[1]

        result = cp.zeros_like(templs_gpu)

        for i in range(n_spectra):
            vsini = vsinis_gpu[i]

            if vsini == 0:
                result[i] = templs_gpu[i]
                continue

            amp = vsini / SPEED_OF_LIGHT
            npts = int(cp.ceil(amp / lnstep))

            # Build rotation kernel
            xgrid = cp.arange(-npts, npts + 1, dtype=cp.float64) * lnstep / amp
            good = cp.abs(xgrid) <= 1

            kernel = cp.zeros_like(xgrid)
            xgrid_good = xgrid[good]

            # Rotation kernel formula
            kernel[good] = ((2 * (1 - eps) * cp.sqrt(1 - xgrid_good**2) +
                            cp.pi / 2 * eps * (1 - xgrid_good**2)) /
                           (2 * cp.pi / (1 - eps / 3)))

            # Convolve using FFT (cupyx.scipy.signal doesn't have convolve in all versions)
            kernel = kernel / cp.sum(kernel)  # Normalize
            # Manual FFT convolution in 'same' mode
            n_kernel = len(kernel)
            n_sig = len(templs_gpu[i])
            # Pad kernel to signal length
            kernel_padded = cp.zeros(n_sig, dtype=cp.float64)
            start_idx = (n_sig - n_kernel) // 2
            kernel_padded[start_idx:start_idx + n_kernel] = kernel
            # FFT convolution
            result[i] = cp.fft.ifft(cp.fft.fft(templs_gpu[i]) * cp.fft.fft(kernel_padded)).real

        return result


def evalRV_batch_gpu(lam_templ, templs, vels, lams, log_step=True, device_id=0):
    """
    Batch evaluate templates at different velocities using GPU spline interpolation

    Parameters
    ----------
    lam_templ : array_like
        Template wavelength array
    templs : array_like, shape (n_templates, n_pixels_templ)
        Template spectra
    vels : array_like, shape (n_spectra,)
        Velocities for each spectrum
    lams : array_like, shape (n_pixels_obs,)
        Observed wavelength array
    log_step : bool
        Whether wavelength is log-spaced
    device_id : int
        GPU device ID

    Returns
    -------
    eval_templs : array_like, shape (n_spectra, n_pixels_obs)
        Evaluated templates
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    with cp.cuda.Device(device_id):
        SPEED_OF_LIGHT = 299792.458  # km/s

        lam_templ_gpu = cp.asarray(lam_templ, dtype=cp.float64)
        templs_gpu = cp.asarray(templs, dtype=cp.float64)
        vels_gpu = cp.asarray(vels, dtype=cp.float64)
        lams_gpu = cp.asarray(lams, dtype=cp.float64)

        n_spectra = len(vels_gpu)
        n_pixels_obs = len(lams_gpu)

        result = cp.zeros((n_spectra, n_pixels_obs), dtype=cp.float64)

        # For each spectrum, create spline and evaluate
        for i in range(n_spectra):
            beta = vels_gpu[i] / SPEED_OF_LIGHT
            shifted_lams = lams_gpu * cp.sqrt((1 - beta) / (1 + beta))

            # Use GPU spline for each template
            spline = CubicSplineGPU(lam_templ_gpu, templs_gpu[i],
                                   log_step=log_step, device_id=device_id)
            result[i] = spline(shifted_lams)

        return result


# Convenience function to check GPU availability
def gpu_available():
    """Check if GPU and CuPy are available"""
    return CUPY_AVAILABLE and cp.cuda.runtime.getDeviceCount() > 0


def get_device_count():
    """Get number of available GPUs"""
    if not CUPY_AVAILABLE:
        return 0
    try:
        return cp.cuda.runtime.getDeviceCount()
    except:
        return 0

#!/usr/bin/env python3
"""
Test script for GPU-accelerated fitting

This script tests the GPU implementations against the CPU versions
to ensure correctness and measure speedup.
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../py'))

from rvspecfit import spec_fit_gpu
from rvspecfit import spec_fit


def test_cubic_spline_gpu():
    """Test GPU cubic spline against numpy version"""
    print("=" * 60)
    print("Testing GPU Cubic Spline Interpolation")
    print("=" * 60)

    # Create test data
    n_knots = 10000
    xs = np.logspace(3, 4, n_knots)
    ys = np.sin(np.log(xs)) + 0.1 * np.random.randn(n_knots)

    # Test points
    n_eval = 50000
    evalx = np.logspace(3, 4, n_eval)

    print(f"Knots: {n_knots}, Evaluation points: {n_eval}")

    # CPU version (using numpy for simplicity - just timing)
    from rvspecfit import spliner
    t0 = time.time()
    spline_cpu = spliner.Spline(xs, ys, log_step=True)
    result_cpu = spline_cpu(evalx)
    t_cpu = time.time() - t0
    print(f"CPU time: {t_cpu:.4f}s")

    # GPU version
    if not spec_fit_gpu.gpu_available():
        print("GPU not available, skipping GPU test")
        return

    try:
        t0 = time.time()
        spline_gpu = spec_fit_gpu.CubicSplineGPU(xs, ys, log_step=True, device_id=0)
        result_gpu = spline_gpu(evalx).get()  # .get() copies to CPU
        t_gpu = time.time() - t0
        print(f"GPU time: {t_gpu:.4f}s")
        print(f"Speedup: {t_cpu/t_gpu:.2f}x")

        # Check accuracy
        max_diff = np.max(np.abs(result_cpu - result_gpu))
        rel_error = max_diff / np.max(np.abs(result_cpu))
        print(f"Max absolute difference: {max_diff:.2e}")
        print(f"Relative error: {rel_error:.2e}")

        if rel_error < 1e-10:
            print("✓ GPU spline test PASSED")
        else:
            print("✗ GPU spline test FAILED")

    except Exception as e:
        print(f"GPU test failed with error: {e}")


def test_chisq_batch():
    """Test batched chi-square computation"""
    print("\n" + "=" * 60)
    print("Testing Batched Chi-Square Computation")
    print("=" * 60)

    n_spectra = 100
    n_pixels = 4000
    n_poly = 10

    # Create synthetic data
    specs = np.random.randn(n_spectra, n_pixels) + 100
    templs = np.random.randn(n_spectra, n_pixels) + 100
    especs = np.ones((n_spectra, n_pixels)) * 10
    polys = np.random.randn(n_poly, n_pixels)

    print(f"Batch size: {n_spectra}, Pixels per spectrum: {n_pixels}")

    # CPU version (single spectrum at a time)
    t0 = time.time()
    chisqs_cpu = []
    for i in range(n_spectra):
        chisq = spec_fit.get_chisq0(specs[i], templs[i], polys,
                                   espec=especs[i])
        chisqs_cpu.append(chisq)
    t_cpu = time.time() - t0
    print(f"CPU time (sequential): {t_cpu:.4f}s ({t_cpu/n_spectra*1000:.2f}ms per spectrum)")

    # GPU version
    if not spec_fit_gpu.gpu_available():
        print("GPU not available, skipping GPU test")
        return

    try:
        import cupy as cp

        t0 = time.time()
        chisqs_gpu, coeffs = spec_fit_gpu.get_chisq0_batch_gpu(
            specs, templs, polys, especs, device_id=0)
        cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
        t_gpu = time.time() - t0

        chisqs_gpu = chisqs_gpu.get()  # Copy to CPU

        print(f"GPU time (batched): {t_gpu:.4f}s ({t_gpu/n_spectra*1000:.2f}ms per spectrum)")
        print(f"Speedup: {t_cpu/t_gpu:.2f}x")

        # Check first few results
        print(f"\nFirst 5 chi-squares (CPU): {chisqs_cpu[:5]}")
        print(f"First 5 chi-squares (GPU): {chisqs_gpu[:5]}")

        max_diff = np.max(np.abs(np.array(chisqs_cpu) - chisqs_gpu))
        rel_error = max_diff / np.mean(np.abs(chisqs_cpu))
        print(f"\nMax absolute difference: {max_diff:.2e}")
        print(f"Relative error: {rel_error:.2e}")

        if rel_error < 1e-8:
            print("✓ GPU batch chi-square test PASSED")
        else:
            print("✗ GPU batch chi-square test FAILED")

    except Exception as e:
        print(f"GPU test failed with error: {e}")
        import traceback
        traceback.print_exc()


def test_gpu_devices():
    """Test GPU device detection"""
    print("\n" + "=" * 60)
    print("GPU Device Information")
    print("=" * 60)

    if spec_fit_gpu.gpu_available():
        n_devices = spec_fit_gpu.get_device_count()
        print(f"Number of GPUs available: {n_devices}")

        try:
            import cupy as cp
            for i in range(n_devices):
                with cp.cuda.Device(i):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    print(f"\nGPU {i}: {props['name'].decode()}")
                    print(f"  Memory: {props['totalGlobalMem'] / 1e9:.1f} GB")
                    print(f"  Compute capability: {props['major']}.{props['minor']}")
        except Exception as e:
            print(f"Error getting device properties: {e}")
    else:
        print("No GPUs available or CuPy not installed")
        print("\nTo install CuPy:")
        print("  pip install cupy-cuda12x  # For CUDA 12.x")
        print("  pip install cupy-cuda11x  # For CUDA 11.x")


if __name__ == '__main__':
    print("GPU Testing Suite for rvspecfit\n")

    # Test GPU availability
    test_gpu_devices()

    # Run tests
    test_cubic_spline_gpu()
    test_chisq_batch()

    print("\n" + "=" * 60)
    print("All tests completed")
    print("=" * 60)

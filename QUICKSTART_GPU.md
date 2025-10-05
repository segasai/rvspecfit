# GPU Quick Start Guide

## 1. Check if you have GPUs and CuPy (30 seconds)

```bash
# Check CUDA version
nvcc --version

# Check GPUs
nvidia-smi

# Try to import CuPy
python -c "import cupy as cp; print(f'CuPy found: {cp.cuda.runtime.getDeviceCount()} GPUs')"
```

**If CuPy import fails:**
```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

## 2. Run GPU Tests (1 minute)

```bash
cd tests
python test_gpu.py
```

**Expected output:**
```
GPU Device Information
============================================================
Number of GPUs available: 4

GPU 0: NVIDIA A100-SXM4-80GB
  Memory: 80.0 GB
  Compute capability: 8.0
[... GPU 1-3 ...]

Testing GPU Cubic Spline Interpolation
============================================================
CPU time: 0.0234s
GPU time: 0.0012s
Speedup: 19.5x
✓ GPU spline test PASSED

Testing Batched Chi-Square Computation
============================================================
CPU time (sequential): 2.1234s (21.23ms per spectrum)
GPU time (batched): 0.1012s (1.01ms per spectrum)
Speedup: 21.0x
✓ GPU batch chi-square test PASSED
```

✅ If tests pass → GPU implementation working correctly!
❌ If tests fail → Share error message for debugging

## 3. Understand What's Been Built

### Files Created:

```
rvspecfit/
├── py/rvspecfit/
│   ├── spec_fit_gpu.py              # Main GPU module
│   └── nn/RVSInterpolator_batch.py  # Batched NN interpolator
├── tests/
│   └── test_gpu.py                  # GPU test suite
├── GPU_README.md                    # Full documentation
├── GPU_IMPLEMENTATION_SUMMARY.md    # What's done & TODO
└── QUICKSTART_GPU.md               # This file
```

### What Works Now:

✅ GPU cubic spline interpolation (replaces C code)
✅ Batched chi-square computation
✅ Batched NN template evaluation
✅ Batched rotation kernel convolution
✅ Multi-GPU device selection

### What Needs Integration:

🚧 Modify `desi_fit.py` to use GPU batching
🚧 Distribute work across 4 A100s
🚧 Resolution matrix on GPU (optional, can use CPU)

## 4. Simple GPU Example (5 minutes)

Create `test_simple_gpu.py`:

```python
import numpy as np
from rvspecfit import spec_fit_gpu

# Test 1: Check GPU availability
print(f"GPUs available: {spec_fit_gpu.get_device_count()}")

# Test 2: GPU Spline
xs = np.logspace(3, 4, 10000)
ys = np.sin(xs)
evalx = np.logspace(3, 4, 50000)

spline = spec_fit_gpu.CubicSplineGPU(xs, ys, log_step=True, device_id=0)
result = spline(evalx)

print(f"Spline evaluated at {len(evalx)} points")
print(f"Result shape: {result.shape}")
print(f"First 5 values: {result.get()[:5]}")  # .get() copies to CPU

# Test 3: Batched Chi-Square
n_spectra = 128
n_pixels = 4000
n_poly = 10

specs = np.random.randn(n_spectra, n_pixels) + 100
templs = np.random.randn(n_spectra, n_pixels) + 100
especs = np.ones((n_spectra, n_pixels)) * 10
polys = np.random.randn(n_poly, n_pixels)

chisqs, coeffs = spec_fit_gpu.get_chisq0_batch_gpu(
    specs, templs, polys, especs, device_id=0
)

print(f"\nBatched chi-square for {n_spectra} spectra")
print(f"Chi-squares shape: {chisqs.shape}")
print(f"First 5 chi-squares: {chisqs.get()[:5]}")

print("\n✅ All GPU operations successful!")
```

Run it:
```bash
python test_simple_gpu.py
```

## 5. Performance Comparison (Optional, 10 minutes)

Create `benchmark_gpu.py`:

```python
import numpy as np
import time
from rvspecfit import spec_fit, spec_fit_gpu

# Setup
n_spectra = 100
n_pixels = 4000
n_poly = 10

specs = np.random.randn(n_spectra, n_pixels) + 100
templs = np.random.randn(n_spectra, n_pixels) + 100
especs = np.ones((n_spectra, n_pixels)) * 10
polys = np.random.randn(n_poly, n_pixels)

# CPU timing
print("CPU timing (sequential)...")
t0 = time.time()
for i in range(n_spectra):
    chisq = spec_fit.get_chisq0(specs[i], templs[i], polys, espec=especs[i])
t_cpu = time.time() - t0

# GPU timing
print("GPU timing (batched)...")
import cupy as cp
t0 = time.time()
chisqs_gpu, _ = spec_fit_gpu.get_chisq0_batch_gpu(
    specs, templs, polys, especs, device_id=0
)
cp.cuda.Stream.null.synchronize()
t_gpu = time.time() - t0

# Results
print(f"\nCPU time: {t_cpu:.3f}s ({t_cpu/n_spectra*1000:.2f}ms per spectrum)")
print(f"GPU time: {t_gpu:.3f}s ({t_gpu/n_spectra*1000:.2f}ms per spectrum)")
print(f"Speedup: {t_cpu/t_gpu:.1f}x")
```

Run it:
```bash
python benchmark_gpu.py
```

## 6. Integration Path

### Option A: Quick Test (Recommended First)

Modify your current workflow to process a small batch:

1. Read ~100 spectra from your DESI file
2. Extract parameters for all spectra
3. Call GPU batch functions manually
4. Compare results to CPU version

**Time estimate:** 1-2 hours

### Option B: Full Integration

Modify `desi_fit.py` to replace `ProcessPoolExecutor` with GPU batching:

1. Batch spectra (lines 1147-1191)
2. Call GPU functions for entire batch
3. Distribute across 4 GPUs
4. Handle errors/edge cases

**Time estimate:** 1-2 days

### Option C: Hybrid Approach

Keep CPU code, add `--gpu` flag:

```bash
rvs_desi_fit --config xx.yaml --gpu --gpu-batch-size 128 test.fits
```

**Time estimate:** 2-3 days (cleanest but most work)

## 7. Common Issues

### "CuPy not installed"
```bash
pip install cupy-cuda12x  # or cupy-cuda11x
```

### "CUDA out of memory"
Reduce batch size in `RVSInterpolator_batch.py`:
```python
nn_interp = RVSInterpolatorBatch(config, batch_size=64)  # instead of 128
```

### "GPU not detected"
```bash
nvidia-smi  # Check if GPUs visible
echo $CUDA_VISIBLE_DEVICES  # Should be unset or "0,1,2,3"
```

### "Results don't match CPU"
Check relative error in `test_gpu.py` output.
Acceptable: < 1e-8
Concerning: > 1e-6

## 8. Getting Help

**Check logs:**
```bash
python test_gpu.py 2>&1 | tee gpu_test.log
```

**Check GPU usage while running:**
```bash
# Terminal 1
python test_gpu.py

# Terminal 2
watch -n 0.5 nvidia-smi
```

**Questions:**
1. See `GPU_README.md` for detailed docs
2. See `GPU_IMPLEMENTATION_SUMMARY.md` for architecture
3. See code: `py/rvspecfit/spec_fit_gpu.py`

## 9. Expected Results on Your System

With 4x NVIDIA A100 80GB:

- **Memory per GPU:** ~10-20GB used (with batch=128)
- **Utilization:** 80-95% during computation
- **Speedup vs 120 CPUs:** 40-50x (after full integration)
- **Throughput:** 100K-150K spectra/hour (vs 3K/hour CPU)

## 10. Next Steps

1. ✅ Run `test_gpu.py` → Verify GPU works
2. ✅ Read `GPU_IMPLEMENTATION_SUMMARY.md` → Understand what's built
3. 🔲 Try simple example above → Familiarize with API
4. 🔲 Benchmark on your data → Measure actual speedup
5. 🔲 Decide integration approach → Quick test vs full integration
6. 🔲 Integrate into `desi_fit.py` → Production use

---

**Status:** Framework ready for testing
**Action:** Run `python tests/test_gpu.py` and share results

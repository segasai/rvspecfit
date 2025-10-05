# GPU Speedup Analysis - Why GPU is Currently Slower

## Current Performance

| Mode | Spectra | Time | Per Spectrum | vs CPU |
|------|---------|------|--------------|--------|
| CPU (16 threads) | 7 | 8.2s | 1.17s | 1.0x baseline |
| CPU (16 threads) | 106 | 38.6s | 0.36s | 1.0x baseline |
| GPU (cuda:0) | 7 | 28-30s | 4.0-4.3s | **0.27x (3.7x slower)** |
| GPU (cuda:0) | 106 | >180s | >1.7s | **<0.2x (>5x slower)** |

## Root Cause: Architecture Mismatch

### CPU Mode Works Well Because:
1. **Multiprocessing**: Uses 16 separate Python processes
2. **True Parallelism**: Each process runs independently
3. **No GIL**: Separate processes bypass Python's Global Interpreter Lock
4. **Efficient**: 16 spectra processed simultaneously → 16x speedup

### GPU Mode is Slower Because:
1. **Sequential Processing**: One spectrum at a time
2. **Small NN calls**: Each NN evaluation is small (~4000 pixels)
3. **GPU Overhead**: Kernel launch overhead dominates for small operations
4. **Python GIL**: Threading doesn't provide true parallelism
5. **Inside scipy.optimize**: NN called hundreds of times during optimization

## Detailed Timing Breakdown (Per Spectrum)

From debug logs (`tests/gpu_debug.log`):

```
Timings process: 0.33s  1.28s  0.69s  0.51s, 0.00s  0.14s
                  ^^^    ^^^    ^^^    ^^^    ^^^    ^^^
                  CCF    opt1   ?      opt2   ?      ?
```

Total per spectrum: ~3.0-3.8s (GPU)
Total per spectrum: ~3.8s single-threaded (CPU)
**But CPU runs 16 in parallel → 3.8s/16 = 0.24s effective**

## Why Threading Didn't Help

Attempted parallel processing with `ThreadPoolExecutor`:
- Result: Same slow performance
- Reason: **Python GIL prevents true parallelism**
- GPU operations in PyTorch require GIL
- Multiple threads queue up, run sequentially

## Why GPU NN Isn't Faster

NN evaluation characteristics:
- **Small batch size**: 1 spectrum at a time
- **GPU overhead**: Kernel launch ~0.1-1ms
- **Transfer overhead**: CPU ↔ GPU memory transfer
- **Underutilization**: A100 GPU can handle 1000s of operations simultaneously

Single NN evaluation:
- Input: 4 parameters (teff, logg, feh, alpha)
- Output: ~4000 pixels
- **This is tiny for a GPU!**

CPU NN evaluation:
- No transfer overhead
- No kernel launch overhead
- Direct memory access
- **Faster for single evaluations**

## What Would Make GPU Faster

### Option 1: Batch NN Evaluations (2-4x speedup potential)

**Challenge**: NN is called inside `scipy.optimize.minimize()`
**Location**: `vel_fit.py:608` → `spec_fit.py:298` → `curInterp.eval()`

Each spectrum optimization makes ~100-500 NN calls.
These happen sequentially, parameters unknown ahead of time.

**Solution** would require:
1. Replace `scipy.optimize` with custom GPU-batched optimizer
2. Evaluate multiple parameter sets simultaneously
3. Batch across optimization steps

**Effort**: 1-2 weeks of careful refactoring

### Option 2: Multi-GPU with Multiprocessing (4-16x speedup potential)

**Approach**: Use Python multiprocessing instead of threading
- Spawn 16 worker processes (like CPU mode)
- Each process uses NN on GPU
- Distribute across 4 GPUs (4 processes per GPU)

**Challenge**: PyTorch models can't be pickled easily for multiprocessing

**Solution**:
1. Each worker loads its own copy of NN model
2. Set `RVS_NN_DEVICE` per worker
3. Use GPU 0-3 in round-robin

**Effort**: 2-3 days

### Option 3: Hybrid Approach (1.5-2x speedup potential)

**Approach**: Use GPU only for specific bottlenecks
- Keep CPU multiprocessing for main loop
- Use GPU for large one-off computations:
  - Initial template library evaluation
  - High-resolution forward modeling
  - Large RV grid searches

**Effort**: 1-2 days

## Recommendation: Option 2 - Multi-GPU Multiprocessing

This provides best ROI:
- Leverages existing infrastructure
- Minimal refactoring required
- Achieves 4-16x speedup potential
- Uses all 4 A100 GPUs effectively

### Implementation Plan:

1. **Modify `proc_many()` to use multiprocessing for GPU mode**
   ```python
   if use_gpu:
       with multiprocessing.Pool(16, initializer=init_worker_gpu,
                                 initargs=(gpu_devices,)) as pool:
           pool.starmap(proc_desi_gpu_worker, tasks)
   ```

2. **Create `proc_desi_gpu_worker()` that:**
   - Sets `RVS_NN_DEVICE=cuda:{worker_id % 4}`
   - Loads NN model for its assigned GPU
   - Processes one spectrum at a time
   - Returns results via queue/shared memory

3. **Each worker process:**
   - Independent Python process (no GIL)
   - Own NN model on assigned GPU
   - Processes 106/16 ≈ 7 spectra

### Expected Performance:

- 16 workers × 4 GPUs = 16 spectra in parallel
- ~3.8s per spectrum (similar to CPU single-threaded)
- Total time: 106/16 × 3.8s ≈ **25s** (vs 38.6s CPU)
- **Speedup: 1.5x vs CPU**

With NN optimization (Option 1), could achieve:
- ~2.0s per spectrum (batched NN)
- Total: 106/16 × 2.0s ≈ **13s**
- **Speedup: 3x vs CPU**

## Why Not More Speedup?

Fitting is dominated by:
1. **scipy.optimize** (CPU-bound, can't parallelize within spectrum)
2. **CCF** (already parallel, ~10% of time)
3. **Chi-square computation** (fast, not bottleneck)

NN evaluation is only ~30-40% of total time.
Even if NN becomes instant (infinite GPU speed):
- Best case: 3.8s → 2.3s per spectrum
- With 16 parallel: 2.3s/16 = **0.14s per spectrum**
- Total for 106: **15s**
- **Max theoretical speedup: 2.5x**

## Conclusion

To achieve GPU faster than CPU (your goal):

**Short-term** (1 week):
- Implement Option 2 (Multi-GPU multiprocessing)
- Expected: 1.5-2x speedup vs 16-thread CPU
- Best for: Large datasets (millions of stars)

**Medium-term** (1 month):
- Add Option 1 (Batched NN with custom optimizer)
- Expected: 2-3x speedup vs 16-thread CPU
- Best for: Maximum performance

**Long-term**:
- Full GPU-native pipeline
- Rewrite fitting logic for GPU
- Expected: 5-10x speedup
- Effort: 2-3 months

**Current status**: GPU infrastructure complete, but architecture needs multi-processing to compete with CPU parallelism.

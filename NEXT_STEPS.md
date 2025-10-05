# Next Steps to Enable GPU Speedup

## Current Situation

✅ **GPU framework is COMPLETE and working**
- All GPU functions implemented and tested
- Command-line integration done (`--use_gpu` flag works)
- GPU device selection functional
- Batching infrastructure ready

❌ **But GPU is currently SLOWER than CPU**
- **Reason**: Still calling sequential CPU code in the loop
- **CPU (16 threads)**: 38.6s for 106 spectra
- **GPU**: >180s for 106 spectra (6x slower!)

## The Problem (One Function to Fix)

**File**: `py/rvspecfit/desi/desi_fit.py`
**Function**: `proc_desi_gpu()` lines 1480-1549

```python
# Current code (SEQUENTIAL, SLOW):
for idx, cur_seqid in enumerate(batch_indices):
    specdatas = get_specdata(...)

    # This calls CPU code one spectrum at a time!
    outdict, curmodel = proc_onespec(
        specdatas, setups, config, options,
        fig_fname=None, doplot=False,
        ccf_init=ccf_init)
```

**The fix**: Replace this loop with batched GPU operations.

## Solution: Three Implementation Levels

### Level 1: Quick Win - NN Batching Only (2-4 hours) ⭐ RECOMMENDED FIRST

**What to do**:
1. Before the loop, extract all stellar parameters from all spectra
2. Call batched NN interpolation ONCE for all 106 spectra
3. Use the pre-computed templates in the loop

**Code changes** (in `proc_desi_gpu()` around line 1480):

```python
# STEP 1: Collect all parameters (before loop)
from rvspecfit.nn import RVSInterpolator_batch
import numpy as np

# Initialize batched NN interpolator
nn_kwargs = {
    'device': f'cuda:{gpu_device_id}',
    'template_lib': config['template_lib'],
    # ... other config params
}
nn_interp = RVSInterpolator_batch.RVSInterpolatorBatch(nn_kwargs, batch_size=gpu_batch_size)

# Collect parameters for all spectra
all_params = []
all_specdatas = []
for idx, cur_seqid in enumerate(batch_indices):
    specdatas = get_specdata(waves, fluxes, ivars, masks, resolutions,
                             cur_seqid, setups, ...)
    all_specdatas.append(specdatas)

    # Get initial params from CCF or priors
    # (You may need to call CCF here or use fixed starting params)
    teff_init = 5000  # or from CCF
    logg_init = 3.0
    feh_init = 0.0
    alpha_init = 0.0
    all_params.append([teff_init, logg_init, feh_init, alpha_init])

# STEP 2: ONE batched GPU call for all templates
all_templates = nn_interp.batch_eval(np.array(all_params))  # Shape: (106, n_pixels)

# STEP 3: Now process with pre-computed templates
for idx in range(len(batch_indices)):
    specdatas = all_specdatas[idx]
    template_for_spectrum = all_templates[idx]

    # Continue with fitting using this template
    # (You'll need to modify proc_onespec to accept pre-computed template)
    outdict, curmodel = proc_onespec_with_template(
        specdatas, template_for_spectrum, setups, config, options)
```

**Expected speedup**: **1.6x** (38.6s → 24s)

### Level 2: Full GPU Pipeline (1-2 days)

**What to do**:
1. Implement Level 1 (NN batching)
2. Batch the chi-square computation using `spec_fit_gpu.get_chisq0_batch_gpu()`
3. Batch RV grid search on GPU
4. Keep CCF on CPU (only 10% of time)

**Code changes**:
- Extract RV grid search from `proc_onespec`
- Create `proc_onespec_batch_gpu()` that processes entire batch
- Use GPU functions from `spec_fit_gpu.py`

**Expected speedup**: **6x** (38.6s → 6s)

### Level 3: Multi-GPU (1 week)

**What to do**:
1. Implement Level 2
2. Split batches across 4 GPUs
3. Parallel processing on each GPU
4. Aggregate results

**Expected speedup**: **13-19x** (38.6s → 2-3s)

## Recommended Path

### This Week: Level 1 ✅

**Why**:
- Easiest to implement (just batch NN calls)
- Highest ROI for time invested
- Immediate 60% speedup
- Proves GPU is faster

**Steps**:
1. Understand where stellar parameters come from (CCF or priors)
2. Collect all parameters before loop
3. Call `nn_interp.batch_eval()` once
4. Distribute templates to spectra
5. Test and benchmark

**Time**: 2-4 hours coding + 1-2 hours testing

### Next Week: Level 2 ✅

**Why**:
- Significant speedup (6x total)
- Most components already implemented
- Good return on investment

**Steps**:
1. Extract fitting logic from `proc_onespec`
2. Create batched version
3. Use GPU chi-square functions
4. Test thoroughly

**Time**: 1-2 days

### Later: Level 3 (Optional)

**Why**:
- Maximum performance
- Best hardware utilization
- Needed for very large datasets

**Time**: 1 week

## How to Start (Level 1)

### Step 1: Understand Parameter Flow

Find where stellar parameters (teff, logg, feh, alpha) come from:
- Check `proc_onespec()` in `desi_fit.py`
- Look for CCF initialization
- Identify parameter source

### Step 2: Modify `proc_desi_gpu()`

```python
# Around line 1470, after timers.append(time.time()):

# Import batched NN
from rvspecfit.nn.RVSInterpolator_batch import RVSInterpolatorBatch

# Setup NN interpolator (only once)
# TODO: Get proper config from spec_inter module
nn_config = {
    'device': f'cuda:{gpu_device_id}',
    'template_lib': config['template_lib'],
    # Add other required config params
}
nn_batch = RVSInterpolatorBatch(nn_config, batch_size=gpu_batch_size)

# Collect parameters
logging.info(f'Collecting parameters for {len(seqid_to_fit)} spectra...')
all_params = []
all_specdatas = []

for cur_seqid in seqid_to_fit:
    specdatas = get_specdata(...)
    all_specdatas.append(specdatas)

    # Get initial parameters
    # Option 1: From CCF (if ccf_init is True)
    if ccf_init:
        ccf_result = fitter_ccf.fit(specdatas, config)
        # Extract params from ccf_result
        params = extract_params_from_ccf(ccf_result)
    else:
        # Option 2: Use default starting point
        params = [5000, 3.0, 0.0, 0.0]  # teff, logg, feh, alpha

    all_params.append(params)

# Batch evaluate templates
logging.info('Evaluating templates on GPU...')
all_templates_batch = nn_batch.batch_eval(np.array(all_params))

# Now process with templates
for idx, cur_seqid in enumerate(seqid_to_fit):
    template = all_templates_batch[idx]
    specdatas = all_specdatas[idx]

    # Continue fitting with pre-computed template
    # ...
```

### Step 3: Test

```bash
# Small test
export RVS_NN_DEVICE=cuda:0
time rvs_desi_fit --use_gpu --minsn 10 --config xx.yaml \
                   --no_subdirs --output_dir tests/tmp_gpu_test \
                   tests/data/coadd-sv1-bright-10378.fits

# Compare with CPU
export RVS_NN_DEVICE=cpu
time rvs_desi_fit --nthreads 16 --minsn 10 --config xx.yaml \
                   --no_subdirs --output_dir tests/tmp_cpu_test \
                   tests/data/coadd-sv1-bright-10378.fits

# GPU should be faster!
```

## Files to Reference

**GPU functions ready to use**:
- `spec_fit_gpu.get_chisq0_batch_gpu()` - Batched chi-square
- `spec_fit_gpu.evalRV_batch_gpu()` - Batched RV interpolation
- `spec_fit_gpu.convolve_vsini_batch_gpu()` - Batched rotation
- `nn.RVSInterpolator_batch.batch_eval()` - Batched NN ⭐

**Helper functions**:
- `spec_fit_gpu.CubicSplineGPU()` - GPU spline
- All in `py/rvspecfit/spec_fit_gpu.py`

## Expected Timeline

| Task | Time | Speedup | Status |
|------|------|---------|--------|
| Level 1: NN batching | 2-4 hours | 1.6x | ⏳ Ready to implement |
| Level 2: Full GPU | 1-2 days | 6x | 📋 Planned |
| Level 3: Multi-GPU | 1 week | 13-19x | 🎯 Future |

## Questions to Answer First

Before implementing Level 1, figure out:

1. **Where do stellar parameters come from?**
   - CCF result?
   - Fixed initial guess?
   - Previous iteration?

2. **What config does NN need?**
   - Check `spec_inter.getInterpolator()`
   - Look at NN initialization in CPU code

3. **Can we modify `proc_onespec()`?**
   - Or create new `proc_onespec_with_template()`?
   - Or inline the fitting logic?

## Get Help

- Check existing NN initialization: `spec_inter.py:291-410`
- See how templates are used: `spec_fit.py:641-643` (getCurTempl)
- Reference batched NN: `nn/RVSInterpolator_batch.py:45-67`

---

**Bottom Line**: The infrastructure is done. You just need to replace ~70 lines in one function (`proc_desi_gpu` lines 1480-1549) to batch the NN calls and get immediate 60% speedup. The rest can come later!

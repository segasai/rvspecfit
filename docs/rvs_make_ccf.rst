rvs_make_ccf
============

The ``rvs_make_ccf`` command creates Fourier-transformed template spectra for fast cross-correlation function (CCF) computation. This is the final step in the rvspecfit template preparation pipeline.

Purpose
-------

This command processes the interpolated template spectra to create FFT-based cross-correlation templates that enable:

- Fast radial velocity measurements via cross-correlation
- Efficient template matching for large spectroscopic surveys
- Support for stellar rotation (v sin i) broadening
- Continuum normalization for cross-correlation

The FFT approach allows computing cross-correlation functions much faster than direct convolution methods.

Basic Usage
-----------

.. code-block:: bash

   rvs_make_ccf --setup config_name \
                --lambda0 4000 --lambda1 9000 \
                --step 1.0 \
                --prefix /path/to/processed/

Command Line Options
--------------------

Required Options
^^^^^^^^^^^^^^^^

``--setup NAME``
    Name of the spectral configuration. Must match the setup used in previous pipeline steps.

``--lambda0 WAVELENGTH``
    Starting wavelength in Angstroms for the cross-correlation region.

``--lambda1 WAVELENGTH``  
    Ending wavelength in Angstroms for the cross-correlation region.

``--step STEP_SIZE``
    Pixel size in Angstroms for the cross-correlation grid.

``--prefix PATH``
    Location of the input spectra (output from ``rvs_make_interpol`` and ``rvs_make_nd``).

Optional Options
^^^^^^^^^^^^^^^^

``--oprefix PATH``
    Location where the output products will be stored. Default: ``'templ_data/'``

``--nthreads N``
    Number of processing threads. Default: ``8``

``--every N``
    Subsample the input template grid by this factor. Only every N-th template will be processed. Default: ``30``

``--vsinis LIST``
    Comma-separated list of v sin i values (in km/s) to include in the CCF template set. Default: no rotation

``--revision STRING``
    Revision identifier for the data files/run.

``--nocontinuum``
    Skip continuum normalization. Creates templates without continuum fitting for cross-correlation.

Advanced Options
^^^^^^^^^^^^^^^^

The wavelength range and step size determine the FFT grid size, which is automatically rounded up to the next power of 2 for efficiency.

Examples
--------

Basic Cross-Correlation Templates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   rvs_make_ccf --setup sdss \
                --lambda0 3500 --lambda1 9500 \
                --step 1.0 \
                --prefix ./processed_templates/

DESI Spectroscopic Survey
^^^^^^^^^^^^^^^^^^^^^^^^^^

Create templates for each DESI channel:

.. code-block:: bash

   # Blue channel
   rvs_make_ccf --setup desi_b \
                --lambda0 3500 --lambda1 5900 \
                --step 0.4 \
                --every 30 \
                --vsinis 0,300 \
                --prefix ./desi_templates/
   
   # Red channel
   rvs_make_ccf --setup desi_r \
                --lambda0 5660 --lambda1 7720 \
                --step 0.4 \
                --every 30 \
                --vsinis 0,300 \
                --prefix ./desi_templates/
   
   # Z channel
   rvs_make_ccf --setup desi_z \
                --lambda0 7420 --lambda1 9924 \
                --step 0.4 \
                --every 30 \
                --vsinis 0,300 \
                --prefix ./desi_templates/

High-Resolution Echelle Spectrograph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   rvs_make_ccf --setup echelle \
                --lambda0 4000 --lambda1 8000 \
                --step 0.1 \
                --every 10 \
                --vsinis 0,5,10,20,50 \
                --prefix ./echelle_templates/ \
                --nthreads 16

Without Continuum Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For applications where continuum normalization is not desired:

.. code-block:: bash

   rvs_make_ccf --setup raw_templates \
                --lambda0 4000 --lambda1 9000 \
                --step 0.8 \
                --nocontinuum \
                --prefix ./templates/

Fast Processing
^^^^^^^^^^^^^^^

For quick testing or when computational resources are limited:

.. code-block:: bash

   rvs_make_ccf --setup test \
                --lambda0 5000 --lambda1 6000 \
                --step 2.0 \
                --every 100 \
                --nthreads 4 \
                --prefix ./templates/

Template Selection and Subsampling
-----------------------------------

The ``--every`` parameter controls template subsampling using a Morton space-filling curve algorithm that:

1. Maps template parameters to a high-dimensional space
2. Sorts templates along the Morton curve
3. Selects every N-th template for uniform sampling

This approach ensures better coverage of parameter space compared to random or sequential sampling.

Typical values:
- ``--every 10``: Dense sampling (slow, high accuracy)
- ``--every 30``: Standard sampling (recommended)
- ``--every 100``: Sparse sampling (fast, lower accuracy)

Stellar Rotation (v sin i) Support
-----------------------------------

The ``--vsinis`` option creates additional templates convolved with rotational broadening:

.. code-block:: bash

   --vsinis 0,10,50,100,300

This creates separate templates for each v sin i value, multiplying the total number of templates. The rotational broadening kernel assumes:
- Linear limb darkening
- Solid-body rotation
- Negligible instrumental broadening compared to rotation

Wavelength Grid and FFT Optimization
-------------------------------------

The command automatically optimizes the FFT grid:

1. Calculates the number of pixels: ``(lambda1 - lambda0) / step``
2. Rounds up to the next power of 2 for FFT efficiency
3. Creates a logarithmic wavelength grid for constant velocity spacing

For example:
- Range: 4000-9000 Å, step: 1.0 Å → 5000 pixels → 8192 FFT points
- Range: 3500-5900 Å, step: 0.4 Å → 6000 pixels → 8192 FFT points

Output Files
------------

The command creates several output files in the specified output directory:

``ccf_<setup>.h5`` or ``ccf_nocont_<setup>.h5``
    HDF5 file containing metadata:
    - ``params``: Template parameters for each FFT template
    - ``ccfconf``: Cross-correlation configuration
    - ``vsinis``: v sin i values for each template
    - ``parnames``: Parameter names
    - ``revision``: Version information

``ccfdat_<setup>.npz`` or ``ccfdat_nocont_<setup>.npz``
    Compressed NumPy file containing:
    - ``fft``: FFT of template spectra
    - ``fft2``: FFT of squared template spectra (for normalization)

``ccfmod_<setup>.npy`` or ``ccfmod_nocont_<setup>.npy``
    NumPy file containing the processed template spectra before FFT transformation.

Processing Pipeline
-------------------

For each selected template spectrum:

1. **Parameter Extraction**: Load stellar parameters from interpolation files
2. **Rotation Convolution**: Apply v sin i broadening if specified
3. **Continuum Fitting**: Fit spline continuum (unless ``--nocontinuum``)
4. **Continuum Normalization**: Divide by continuum fit
5. **Wavelength Interpolation**: Interpolate to CCF wavelength grid
6. **FFT Computation**: Compute FFT and FFT of squared spectrum
7. **Storage**: Save to output files

Continuum Normalization Details
-------------------------------

When continuum normalization is enabled (default):

1. **Spline Fitting**: Fits a smooth spline to the spectrum
2. **Node Spacing**: Automatic node spacing based on wavelength range
3. **Robust Fitting**: Uses robust regression to handle absorption lines
4. **Normalization**: Divides spectrum by continuum fit

The continuum fitting parameters are automatically optimized based on the wavelength range and spectral resolution.

Memory and Performance
----------------------

**Memory Usage:**
- Depends on template grid size and FFT dimensions
- Typical usage: 1-10 GB for large surveys
- Scales with: number_templates × number_wavelength_points × number_vsinis

**Processing Time:**
- Scales with number of templates and wavelength points
- FFT computation is the most expensive step
- Parallelized across templates using ``--nthreads``

**Storage Requirements:**
- FFT files can be large (100s of MB to GB)
- Storage scales with template count and wavelength range

Quality Control
---------------

To verify CCF template quality:

1. **Coverage Check**: Ensure templates span the required parameter space
2. **Sampling Uniformity**: Verify ``--every`` provides adequate sampling
3. **Wavelength Range**: Confirm coverage matches observational data
4. **Continuum Quality**: Check continuum fits for representative templates

Integration with rvspecfit
--------------------------

The CCF templates are used by:
- ``rvspecfit`` main fitting routines for radial velocity determination
- Cross-correlation analysis tools
- Template matching algorithms

The FFT-based approach enables:
- Fast cross-correlation computation (O(N log N) vs O(N²))
- Real-time radial velocity measurements
- Efficient processing of large spectroscopic datasets

Troubleshooting
---------------

**"No such file" errors**
    Ensure ``rvs_make_interpol`` and ``rvs_make_nd`` have been run successfully.

**Memory errors**
    Reduce ``--nthreads``, increase ``--every``, or process smaller wavelength ranges.

**FFT size warnings**
    Large FFT grids may be inefficient. Consider adjusting ``--step`` to optimize grid size.

**Continuum fitting failures**
    Some templates may have poor continuum fits. Check individual templates or use ``--nocontinuum``.

**v sin i convolution issues**
    Very high rotation rates may require careful wavelength sampling. Ensure adequate resolution.

See Also
--------

- :doc:`rvs_make_nd` - Previous step: create n-dimensional interpolation
- :doc:`rvs_make_interpol` - Create processed template spectra  
- :doc:`rvs_read_grid` - First step: create template database
rvs_make_interpol
================

The ``rvs_make_interpol`` command processes stellar template spectra by convolving them to a target resolution and rebinning them to a common wavelength grid. This is the second step in the rvspecfit template preparation pipeline.

Purpose
-------

This command takes the template database created by ``rvs_read_grid`` and processes all templates to:

- Convolve spectra to a target instrumental resolution
- Rebin to a common wavelength grid (linear or logarithmic)
- Normalize by continuum (optional)
- Convert to logarithmic flux space
- Save processed templates for interpolation

The output is used by ``rvs_make_nd`` to create interpolation grids.

Basic Usage
-----------

.. code-block:: bash

   rvs_make_interpol --setup config_name \
                     --lambda0 4000 --lambda1 9000 \
                     --resol 2000 --step 1.0 \
                     --templdb files.db \
                     --templprefix /path/to/templates/ \
                     --wavefile wave.fits

Command Line Options
--------------------

Required Options
^^^^^^^^^^^^^^^^

``--setup NAME``
    Name of the spectral configuration. This identifier is used in output filenames and must be unique for each setup.

``--lambda0 WAVELENGTH``
    Start wavelength of the new grid in Angstroms.

``--lambda1 WAVELENGTH``
    End wavelength of the new grid in Angstroms.

``--step STEP_SIZE``
    Pixel size in Angstroms. For log-spacing (default), this corresponds to the pixel size at the shortest wavelengths.

``--templdb FILENAME``
    Path to the SQLite database created by ``rvs_read_grid``.

``--templprefix PATH``
    Path to the template spectra files.

``--wavefile FILENAME``
    Path to the FITS file containing the wavelength grid of the original templates.

Resolution Options (choose one)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--resol RESOLUTION``
    Constant spectral resolution R = λ/Δλ for the output grid.

``--resol_func FUNCTION``
    Spectral resolution as a function of wavelength. Must be a string expression using 'x' as the wavelength variable.
    
    Examples:
    - ``'2000'`` - constant resolution of 2000
    - ``'1000 + 2*x'`` - linear function of wavelength
    - ``'x/1.55'`` - resolution proportional to wavelength (constant FWHM)

Additional Options
^^^^^^^^^^^^^^^^^^

``--revision STRING``
    Revision identifier for the templates. Default: empty string

``--parameter_names LIST``
    Comma-separated list of parameter names for interpolation. Default: ``'teff,logg,feh,alpha'``

``--log_parameters INDICES``
    Comma-separated list of parameter indices to take log₁₀ of during interpolation. Default: ``'0'`` (log of first parameter, typically Teff)

``--oprefix PATH``
    Output directory where processed templates will be created. Default: ``'templ_data/'``

``--resolution0 RESOLUTION``
    Resolution of the input template grid. Default: ``100000``

``--nthreads N``
    Number of processing threads. Default: ``8``

Wavelength Grid Options
^^^^^^^^^^^^^^^^^^^^^^^

``--log`` / ``--no-log``
    Generate spectra on logarithmic wavelength scale. Default: ``--log``

``--air``
    Generate spectra in air wavelengths (rather than vacuum). Default: vacuum

Normalization Options
^^^^^^^^^^^^^^^^^^^^^

``--normalize`` / ``--no-normalize``
    Normalize spectra by continuum fit. Default: ``--normalize``

``--fixed_fwhm``
    Make the FWHM of the line spread function constant rather than constant resolution R = λ/Δλ.

Examples
--------

Basic SDSS-like Setup
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   rvs_make_interpol --setup sdss \
                     --lambda0 3500 --lambda1 9500 \
                     --resol 2000 --step 1.0 \
                     --templdb templates.db \
                     --templprefix /data/PHOENIX/ \
                     --wavefile WAVE_PHOENIX.fits \
                     --oprefix ./processed_templates/

DESI Blue Channel
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   rvs_make_interpol --setup desi_b \
                     --lambda0 3500 --lambda1 5900 \
                     --resol_func 'x/1.55' \
                     --step 0.4 \
                     --templdb templates.db \
                     --templprefix /data/PHOENIX/ \
                     --wavefile WAVE_PHOENIX.fits

High-Resolution Setup
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   rvs_make_interpol --setup hires \
                     --lambda0 4000 --lambda1 8000 \
                     --resol 50000 --step 0.1 \
                     --templdb templates.db \
                     --templprefix /data/PHOENIX/ \
                     --wavefile WAVE_PHOENIX.fits \
                     --resolution0 500000

Custom Parameters
^^^^^^^^^^^^^^^^^

For templates with additional parameters like microturbulence:

.. code-block:: bash

   rvs_make_interpol --setup custom \
                     --lambda0 4000 --lambda1 9000 \
                     --resol 3000 --step 0.8 \
                     --parameter_names 'teff,logg,feh,alpha,vmic' \
                     --log_parameters '0' \
                     --templdb templates.db \
                     --templprefix /data/templates/ \
                     --wavefile wave.fits

Variable Resolution Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For instruments with wavelength-dependent resolution:

.. code-block:: bash

   rvs_make_interpol --setup variable_res \
                     --lambda0 3500 --lambda1 9500 \
                     --resol_func '1000 + 0.5*x' \
                     --step 1.2 \
                     --templdb templates.db \
                     --templprefix /data/PHOENIX/ \
                     --wavefile wave.fits

Output Files
------------

The command creates several files in the output directory:

``specs_<setup>.h5``
    HDF5 file containing:
    - ``specs``: Processed spectra array (N_templates × N_wavelengths)
    - ``vec``: Parameter vectors for each template
    - ``lam``: Common wavelength grid
    - ``parnames``: Parameter names
    - ``lognorms``: Normalization factors
    - Metadata about processing parameters

Processing Steps
----------------

1. **Template Loading**: Load each template spectrum from the database
2. **Resolution Convolution**: Convolve to target resolution using Gaussian kernel
3. **Wavelength Conversion**: Convert from vacuum to air wavelengths if requested
4. **Rebinning**: Interpolate to common wavelength grid
5. **Continuum Normalization**: Fit and divide by linear continuum
6. **Logarithmic Conversion**: Take log of normalized flux
7. **Storage**: Save processed spectra and metadata

Resolution Convolution Details
------------------------------

The resolution convolution assumes the input templates have a resolution specified by ``--resolution0``. The convolution kernel σ is calculated as:

.. math::

   \sigma = \frac{\lambda}{\text{FWHM}_{\text{conv}} \times 2\sqrt{2\ln 2}}

where:

.. math::

   \text{FWHM}_{\text{conv}} = \sqrt{\text{FWHM}_{\text{target}}^2 - \text{FWHM}_{\text{input}}^2}

Wavelength Grid Options
-----------------------

**Linear Grid** (``--no-log``):
    Constant wavelength step: λₙ = λ₀ + n × step

**Logarithmic Grid** (``--log``, default):
    Constant velocity step: ln(λₙ) = ln(λ₀) + n × step_log
    
    The logarithmic step is calculated to give the specified linear step at the center of the wavelength range.

Memory and Performance
----------------------

- Processing time scales with number of templates × number of wavelength points
- Memory usage depends on grid size and number of threads
- Use ``--nthreads`` to control parallelization
- Large grids may require substantial RAM (several GB for full PHOENIX grid)

Troubleshooting
---------------

**"Cannot generate the spectra as the wavelength range..."**
    The requested wavelength range extends beyond the template wavelength coverage. Reduce ``--lambda0`` and/or ``--lambda1``.

**"The spectrum is not finite (has nans or infs)"**
    A template spectrum contains invalid values. Check the input templates or exclude problematic spectra from the database.

**Resolution errors**
    Ensure ``--resolution0`` matches the actual resolution of input templates. The target resolution must be lower than the input resolution.

**Memory errors**
    Reduce ``--nthreads`` or process smaller wavelength ranges separately.

See Also
--------

- :doc:`rvs_read_grid` - Previous step: create template database
- :doc:`rvs_make_nd` - Next step: create n-dimensional interpolation
- :doc:`rvs_make_ccf` - Create cross-correlation functions
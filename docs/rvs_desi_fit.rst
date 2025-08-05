rvs_desi_fit
============

The ``rvs_desi_fit`` command fits DESI spectra using the rvspecfit pipeline to determine radial velocities and stellar atmospheric parameters. This is the main interface for processing DESI spectroscopic data.

Purpose
-------

This command is specifically designed for the Dark Energy Spectroscopic Instrument (DESI) survey data and provides:

- Automated fitting of stellar spectra from DESI coadded files
- Radial velocity and stellar parameter determination
- Integration with DESI data formats and metadata
- Support for parallel processing and MPI
- Quality control and warning flags
- Integration with redrock/zbest results

The command processes DESI coadd FITS files and produces tables with fitted parameters, quality metrics, and optional diagnostic plots.

Basic Usage
-----------

.. code-block:: bash

   rvs_desi_fit --config config.yaml --output_dir ./results/ spectrum.fits

Command Line Options
--------------------

Input Files and Selection
^^^^^^^^^^^^^^^^^^^^^^^^^

**Positional Arguments**

``input_files``
    Space-separated list of DESI coadd FITS files to process.

**File Input Options**

``--input_file_from FILENAME``
    Read the list of spectral files from a text file (one filename per line).

``--queue_file``
    Treat the input file list as a queue where entries are deleted as soon as they're processed. Useful for distributed processing.

Target Selection
^^^^^^^^^^^^^^^^

``--targetid TARGETID``
    Fit only a specific target ID. Useful for debugging or focused analysis.

``--targetid_file_from FILENAME``
    Fit only target IDs listed in the specified file (one target ID per line).

``--minsn THRESHOLD``
    Fit only spectra with signal-to-noise ratio larger than this threshold. Default: ``-1e9`` (no filtering)

``--minexpid EXPID``
    Process only exposures with exposure ID >= this value.

``--maxexpid EXPID``
    Process only exposures with exposure ID <= this value.

``--objtypes TYPES``
    Comma-separated list of object types to fit. Examples: ``'MWS_ANY,SCND_ANY,STD_*'``. Uses DESI target selection bitmasks.

Redrock Integration
^^^^^^^^^^^^^^^^^^^

``--zbest_select``
    Use the zbest/redrock file to fit only potentially interesting targets (e.g., stars vs galaxies).

``--zbest_include``
    Include zbest/redrock information in the output table without using it for target selection.

Processing Options
^^^^^^^^^^^^^^^^^^

``--config FILENAME``
    Path to the YAML configuration file specifying template library location and fitting parameters.

``--nthreads N``
    Number of threads for parallel processing. Default: ``1``

``--npoly N``
    Number of polynomial coefficients for continuum fitting. Overrides config file setting if specified.

``--fitarm ARMS``
    Comma-separated list of spectral arms to fit. Options: ``'b'``, ``'r'``, ``'z'``, or combinations like ``'b,r,z'``.

``--param_init METHOD``
    Method for initializing fit parameters and radial velocity. Default: ``'CCF'`` (cross-correlation function)

Resolution Matrix
^^^^^^^^^^^^^^^^^

``--resolution_matrix``
    Use DESI resolution matrix in fitting for more accurate results.

``--no-resolution_matrix``
    Do not use resolution matrix (faster but less accurate). Default behavior.

Output Control
^^^^^^^^^^^^^^

``--output_dir PATH``
    Output directory for result tables. Default: ``'.'``

``--output_tab_prefix PREFIX``
    Prefix for output table files. Default: ``'rvtab'``

``--output_mod_prefix PREFIX``
    Prefix for output model files. Default: ``'rvmod'``

``--subdirs`` / ``--no_subdirs``
    Create/don't create subdirectories in output directory. Default: create subdirectories

File Handling
^^^^^^^^^^^^^

``--overwrite``
    Overwrite existing output products. Otherwise, attempt to update/append.

``--skipexisting``
    Skip processing if output products already exist.

Diagnostics and Plotting
^^^^^^^^^^^^^^^^^^^^^^^^^

``--doplot``
    Generate diagnostic plots showing spectral fits.

``--figure_dir PATH``
    Directory for fit figures. Default: ``'./'``

``--figure_prefix PREFIX``
    Prefix for figure filenames. Default: ``'fig'``

Cross-Correlation Options
^^^^^^^^^^^^^^^^^^^^^^^^^

``--ccf_continuum_normalize`` / ``--no_ccf_continuum_normalize``
    Enable/disable continuum normalization during cross-correlation. Default: enabled

Logging and Monitoring
^^^^^^^^^^^^^^^^^^^^^^

``--log FILENAME``
    Write log messages to specified file. For MPI runs, use ``%d`` to include rank in filename.

``--log_level LEVEL``
    Set logging level. Options: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``. Default: ``WARNING``

``--process_status_file FILENAME``
    File to record successfully processed files for monitoring progress.

Advanced Options
^^^^^^^^^^^^^^^^

``--mpi``
    Use MPI for distributed processing across multiple nodes.

``--throw_exceptions``
    Don't protect against exceptions inside rvspecfit (useful for debugging).

``--version``
    Display software version and exit.

Examples
--------

Basic DESI Fitting
^^^^^^^^^^^^^^^^^^

Process a single DESI coadd file:

.. code-block:: bash

   rvs_desi_fit --config desi_config.yaml \
                --output_dir ./desi_results/ \
                coadd-sv1-bright-10378.fits

Batch Processing
^^^^^^^^^^^^^^^^

Process multiple files with parallel processing:

.. code-block:: bash

   rvs_desi_fit --config desi_config.yaml \
                --output_dir ./desi_results/ \
                --nthreads 8 \
                --process_status_file processing.log \
                coadd-*.fits

High S/N Stars Only
^^^^^^^^^^^^^^^^^^^

Fit only high signal-to-noise stellar targets:

.. code-block:: bash

   rvs_desi_fit --config desi_config.yaml \
                --output_dir ./stellar_results/ \
                --minsn 10 \
                --objtypes 'MWS_ANY,SCND_ANY' \
                --zbest_select \
                coadd-sv1-bright-10378.fits

Specific Target Analysis
^^^^^^^^^^^^^^^^^^^^^^^^

Analyze specific targets with diagnostic plots:

.. code-block:: bash

   echo "39628422527323686" > targets.txt
   echo "39628422531515839" >> targets.txt
   
   rvs_desi_fit --config desi_config.yaml \
                --output_dir ./target_analysis/ \
                --targetid_file_from targets.txt \
                --doplot \
                --figure_dir ./plots/ \
                --log_level DEBUG \
                coadd-sv1-bright-10378.fits

Single Arm Processing
^^^^^^^^^^^^^^^^^^^^^

Fit only the blue arm spectra:

.. code-block:: bash

   rvs_desi_fit --config desi_config.yaml \
                --output_dir ./blue_arm_results/ \
                --fitarm b \
                --resolution_matrix \
                coadd-sv1-bright-10378.fits

Queue-Based Processing
^^^^^^^^^^^^^^^^^^^^^^

Use a file queue for distributed processing:

.. code-block:: bash

   ls /data/desi/coadd-*.fits > file_queue.txt
   
   rvs_desi_fit --config desi_config.yaml \
                --output_dir ./queue_processing/ \
                --input_file_from file_queue.txt \
                --queue_file \
                --mpi \
                --nthreads 4

Configuration File
------------------

The command requires a YAML configuration file specifying template library paths and fitting parameters:

.. code-block:: yaml

   # desi_config.yaml
   template_lib: '/path/to/desi_templates/'
   min_vel: -1000      # km/s
   max_vel: 1000       # km/s  
   min_vel_step: 0.2   # km/s
   vel_step0: 5        # km/s
   min_vsini: 0.1      # km/s
   max_vsini: 500      # km/s

Output Files
------------

Result Tables
^^^^^^^^^^^^^

``rvtab_<filename>.fits`` (or custom prefix)
    Main results table with columns:
    
    - **TARGETID**: DESI target identifier
    - **VRAD**: Radial velocity [km/s]
    - **VRAD_ERR**: Radial velocity uncertainty [km/s]
    - **TEFF**: Effective temperature [K]
    - **TEFF_ERR**: Temperature uncertainty [K]  
    - **LOGG**: Surface gravity [dex]
    - **LOGG_ERR**: Surface gravity uncertainty [dex]
    - **FEH**: Metallicity [Fe/H] [dex]
    - **FEH_ERR**: Metallicity uncertainty [dex]
    - **ALPHA**: Alpha enhancement [α/Fe] [dex]
    - **ALPHA_ERR**: Alpha enhancement uncertainty [dex]
    - **VSINI**: Rotational velocity [km/s]
    - **VSINI_ERR**: Rotational velocity uncertainty [dex]
    - **CHISQ_TOT**: Total χ² of the fit
    - **RVS_WARN**: Quality warning bitmask
    - Additional DESI metadata columns

Model Spectra
^^^^^^^^^^^^^

``rvmod_<filename>.fits`` (or custom prefix)
    Best-fit model spectra for comparison with observations.

Quality Flags
-------------

The ``RVS_WARN`` bitmask indicates potential issues:

- **1 (CHISQ_WARN)**: χ² vs continuum is suspiciously large
- **2 (RV_WARN)**: Radial velocity too close to search boundaries
- **4 (RVERR_WARN)**: Radial velocity uncertainty too large  
- **8 (PARAM_WARN)**: Stellar parameters too close to grid boundaries
- **16 (VSINI_WARN)**: v sin i value suspiciously large
- **32 (BAD_SPECTRUM)**: Problem with spectrum itself
- **64 (BAD_HESSIAN)**: Issue with error estimation (Hessian matrix)

DESI Data Integration
--------------------

File Format Support
^^^^^^^^^^^^^^^^^^^

- Reads DESI coadd FITS files with standard extensions
- Handles DESI fibermap information for target metadata
- Supports DESI spectral resolution matrices
- Compatible with DESI data model conventions

Target Selection
^^^^^^^^^^^^^^^^

Uses DESI target selection bitmasks:

- **MWS_ANY**: Milky Way Survey targets
- **SCND_ANY**: Secondary targets  
- **STD_***: Standard stars (various types)

Object types are filtered using DESI ``OBJTYPE`` and ``DESI_TARGET`` information.

Redrock Integration
^^^^^^^^^^^^^^^^^^^

When ``--zbest_select`` or ``--zbest_include`` are used:

- Reads redrock/zbest files associated with input spectra
- Can filter targets based on redrock classifications
- Includes redrock redshifts and spectral types in output

Performance Considerations
--------------------------

**Single Node Processing:**
- Use ``--nthreads`` to parallelize across CPU cores
- Typical performance: ~10-100 spectra per minute per core
- Memory usage: ~1-2 GB per thread

**Multi-Node Processing:**
- Use ``--mpi`` for distributed processing
- Requires MPI-enabled environment  
- Use ``--queue_file`` with shared file system for work distribution
- Log files should include MPI rank: ``--log logfile_%d.log``

**Resolution Matrix:**
- ``--resolution_matrix`` improves accuracy but increases compute time by ~2-3x
- Recommended for final science analysis
- May not be necessary for initial target screening

Troubleshooting
---------------

**File Access Issues**
    Ensure DESI files are accessible and properly formatted. Check file permissions and paths.

**Template Library Problems**
    Verify config file points to correct template library. Ensure all required template files (ccf, interp, etc.) are present.

**Memory Issues**
    Reduce ``--nthreads`` or process smaller batches. High-resolution fits require substantial memory.

**MPI Problems**
    Check MPI installation and network configuration. Ensure shared filesystem access across nodes.

**Target Selection Issues**
    Verify ``--objtypes`` matches DESI target selection. Check that zbest files exist if using zbest options.

**Quality Warnings**
    High ``RVS_WARN`` values indicate potential fit issues. Check individual spectra and consider filtering criteria.

Integration with DESI Pipeline
-------------------------------

The command integrates with DESI data processing workflows:

1. **Input**: DESI coadd spectra from spectroscopic pipeline
2. **Processing**: rvspecfit analysis with DESI-specific adaptations
3. **Output**: FITS tables compatible with DESI data model
4. **Quality**: Warning flags following DESI conventions

Results can be ingested into DESI databases or used for science analysis following DESI data access protocols.

See Also
--------

- :doc:`rvs_read_grid` - Prepare template database
- :doc:`rvs_make_interpol` - Create interpolated templates  
- :doc:`rvs_make_ccf` - Generate cross-correlation templates
rvs_make_nd
===========

The ``rvs_make_nd`` command creates n-dimensional interpolation structures from the processed template spectra. This is the third step in the rvspecfit template preparation pipeline.

Purpose
-------

This command takes the processed template spectra created by ``rvs_make_interpol`` and builds interpolation structures that allow fast computation of synthetic spectra at arbitrary points in parameter space (Teff, log g, [Fe/H], [α/Fe], etc.).

The command supports two interpolation methods:
- **Delaunay triangulation** (default): Flexible for irregular grids
- **Regular grid interpolation**: More efficient for structured grids

Basic Usage
-----------

.. code-block:: bash

   rvs_make_nd --setup config_name --prefix /path/to/processed/

Command Line Options
--------------------

Required Options
^^^^^^^^^^^^^^^^

``--setup NAME``
    Name of the spectral configuration. Must match the setup name used in ``rvs_make_interpol``.

``--prefix PATH``
    Location of the interpolated and convolved input spectra created by ``rvs_make_interpol``.

Optional Options
^^^^^^^^^^^^^^^^

``--regulargrid``
    Use regular grid interpolation instead of Delaunay triangulation. Only use this if your template grid is perfectly regular (rectangular grid in parameter space).

``--revision STRING``
    Revision identifier for the data files/run. Used for tracking different versions.

Interpolation Methods
---------------------

Delaunay Triangulation (Default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method works with irregularly spaced template grids by:

1. Creating a Delaunay triangulation of the parameter space
2. Adding "ghost" points outside the convex hull for extrapolation
3. Assigning nearest-neighbor spectra to ghost points
4. Enabling interpolation within the convex hull and controlled extrapolation outside

**Advantages:**
- Works with any grid structure
- Handles missing grid points gracefully
- Provides controlled extrapolation

**Disadvantages:**
- Slightly slower than regular grid interpolation
- More complex internal structure

Regular Grid Interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method assumes the templates lie on a regular rectangular grid in parameter space:

1. Identifies unique values along each parameter dimension
2. Creates a regular grid structure
3. Uses multi-linear interpolation

**Advantages:**
- Faster interpolation
- Lower memory usage
- Simpler structure

**Disadvantages:**
- Only works with perfectly regular grids
- No extrapolation capability
- Fails if any grid points are missing

Examples
--------

Basic Usage (Delaunay Triangulation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   rvs_make_nd --setup sdss \
               --prefix ./processed_templates/

This creates interpolation files for the "sdss" configuration using Delaunay triangulation.

Regular Grid
^^^^^^^^^^^^

For a perfectly regular template grid:

.. code-block:: bash

   rvs_make_nd --setup regular_config \
               --prefix ./processed_templates/ \
               --regulargrid

With Revision Tracking
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   rvs_make_nd --setup desi_b \
               --prefix ./desi_templates/ \
               --revision "v2.1_phoenix"

Multiple Configurations
^^^^^^^^^^^^^^^^^^^^^^^^

Process multiple spectral setups:

.. code-block:: bash

   # Process blue channel
   rvs_make_nd --setup desi_b --prefix ./templates/
   
   # Process red channel  
   rvs_make_nd --setup desi_r --prefix ./templates/
   
   # Process z-band channel
   rvs_make_nd --setup desi_z --prefix ./templates/

Input Files
-----------

The command reads files created by ``rvs_make_interpol``:

``specs_<setup>.h5``
    HDF5 file containing processed template spectra and metadata

Output Files
------------

The command creates two main output files:

``interp_<setup>.h5``
    HDF5 file containing the interpolation structure:
    
    **For Delaunay triangulation:**
    - ``triang``: Delaunay triangulation object
    - ``extraflags``: Flags indicating extrapolation regions
    - ``vec``: Parameter vectors (including ghost points)
    - ``lam``: Wavelength grid
    - ``parnames``: Parameter names
    - ``lognorms``: Normalization factors
    - Metadata
    
    **For regular grid:**
    - ``uvecs``: Unique parameter values for each dimension
    - ``idgrid``: Grid mapping from parameter indices to spectrum indices
    - ``vec``: Parameter vectors
    - ``lam``: Wavelength grid
    - ``parnames``: Parameter names
    - ``lognorms``: Normalization factors
    - Metadata

``interpdat_<setup>.npy``
    NumPy binary file containing the actual template spectra in contiguous array format for fast access.

Grid Augmentation (Delaunay Method)
-----------------------------------

For the Delaunay triangulation method, the command automatically augments the template grid:

1. **Edge Detection**: Finds the convex hull of the parameter space
2. **Ghost Point Creation**: Creates additional points outside the convex hull
3. **Nearest Neighbor Assignment**: Assigns the nearest template spectrum to each ghost point
4. **Perturbation**: Slightly perturbs grid points to avoid numerical instabilities

This ensures that interpolation requests slightly outside the original grid still return reasonable results rather than failing.

Performance Characteristics
---------------------------

**Delaunay Triangulation:**
- Setup time: O(N log N) where N is number of templates
- Memory: ~2× template data size
- Interpolation speed: O(log N) per spectrum

**Regular Grid:**
- Setup time: O(N)
- Memory: ~1.5× template data size  
- Interpolation speed: O(1) per spectrum

Memory Usage
------------

Typical memory requirements:
- Small grid (1000 templates): ~100 MB
- Medium grid (10,000 templates): ~1 GB
- Large grid (100,000 templates): ~10 GB

The exact memory usage depends on:
- Number of templates
- Number of wavelength points
- Number of parameters
- Interpolation method chosen

Troubleshooting
---------------

**"No such file or directory" for specs_<setup>.h5**
    Run ``rvs_make_interpol`` first to create the input files.

**"Something is broken the parameters are not finite"**
    The template parameters contain NaN or infinite values. Check the original template database.

**Regular grid interpolation fails**
    Your grid is not perfectly regular. Use the default Delaunay triangulation method instead.

**Memory errors**
    The template grid is too large for available memory. Consider:
    - Processing subsets of the parameter space
    - Using a machine with more RAM
    - Reducing the wavelength range or resolution

**Triangulation instabilities**
    Rarely, the Delaunay triangulation may fail due to numerical precision issues. This is mitigated by automatic point perturbation, but very pathological grids might still cause problems.

Quality Assessment
------------------

To verify the interpolation quality:

1. Check that all template parameters are covered
2. Ensure no large gaps exist in parameter space
3. Test interpolation at known grid points
4. Verify extrapolation behavior at grid edges

The interpolation quality depends on:
- Template grid density and coverage
- Parameter space dimensionality
- Smoothness of spectral variations

Integration with rvspecfit
--------------------------

The interpolation files created by this command are used by:
- ``rvs_make_ccf`` for creating cross-correlation templates
- The main rvspecfit fitting routines for generating synthetic spectra during χ² minimization

See Also
--------

- :doc:`rvs_make_interpol` - Previous step: process template spectra
- :doc:`rvs_make_ccf` - Next step: create cross-correlation functions
- :doc:`rvs_read_grid` - First step: create template database
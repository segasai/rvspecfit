rvs_read_grid
=============

The ``rvs_read_grid`` command creates an SQLite database catalog of stellar template spectra, which is the first step in the rvspecfit template preparation pipeline.

Purpose
-------

This command scans a directory of FITS files containing stellar template spectra (typically PHOENIX models) and creates a database that catalogs all the templates with their atmospheric parameters (Teff, log g, [Fe/H], [α/Fe], etc.). The database is used by subsequent commands in the pipeline.

Basic Usage
-----------

.. code-block:: bash

   rvs_read_grid --prefix /path/to/templates/ --templdb files.db

Command Line Options
--------------------

Required Options
^^^^^^^^^^^^^^^^

``--prefix PATH``
    The location of the input template grid directory. This should contain the FITS files with stellar template spectra.

``--templdb FILENAME``
    The filename where the SQLite database describing the template library will be stored. Default: ``files.db``

Header Keyword Mapping Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--keyword_teff KEYWORD``
    The FITS header keyword for effective temperature. Default: ``PHXTEFF``

``--keyword_logg KEYWORD``
    The FITS header keyword for surface gravity (log g). Default: ``PHXLOGG``

``--keyword_metallicity KEYWORD``
    The FITS header keyword for metallicity [Fe/H]. Default: ``PHXM_H``

``--keyword_alpha KEYWORD``
    The FITS header keyword for alpha enhancement [α/Fe]. Default: ``PHXALPHA``

Parameter Naming Options
^^^^^^^^^^^^^^^^^^^^^^^^

``--name_metallicity NAME``
    The internal name for the metallicity parameter. Default: ``feh``

``--name_alpha NAME``
    The internal name for the alpha enhancement parameter. Default: ``alpha``

File Selection Options
^^^^^^^^^^^^^^^^^^^^^^

``--glob_mask PATTERN``
    The glob pattern to find the spectral FITS files. Default: ``*/*fits``

Additional Parameters
^^^^^^^^^^^^^^^^^^^^^

``--extra_params PARAM_LIST``
    Extra template parameters to store in the database. Format: comma-separated list of ``internal_name:FITS_keyword`` pairs.
    
    Example: ``--extra_params vmic:VMIC,vsini:VSINI`` would store microturbulence and rotational velocity if those keywords exist in the FITS headers.

Examples
--------

Basic Usage with PHOENIX Templates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   rvs_read_grid --prefix /data/PHOENIX-ACES-AGSS-COND-2011/ \
                 --templdb phoenix_templates.db

Custom Header Keywords
^^^^^^^^^^^^^^^^^^^^^^

For templates with non-standard FITS keywords:

.. code-block:: bash

   rvs_read_grid --prefix /data/custom_templates/ \
                 --templdb custom.db \
                 --keyword_teff TEMP \
                 --keyword_logg GRAVITY \
                 --keyword_metallicity MH \
                 --keyword_alpha ALPHA_FE

Including Extra Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

To store additional parameters like microturbulence velocity:

.. code-block:: bash

   rvs_read_grid --prefix /data/PHOENIX-ACES-AGSS-COND-2011/ \
                 --templdb phoenix_with_vmic.db \
                 --extra_params vmic:VMIC

Different File Organization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For templates organized differently:

.. code-block:: bash

   rvs_read_grid --prefix /data/templates/ \
                 --templdb templates.db \
                 --glob_mask "*.fits"

Output
------

The command creates an SQLite database with two main tables:

1. **files** table: Contains one row per template spectrum with:
   - ``filename``: Path to the FITS file (relative to prefix)
   - ``teff``: Effective temperature
   - ``logg``: Surface gravity
   - ``feh`` (or custom name): Metallicity
   - ``alpha`` (or custom name): Alpha enhancement
   - Additional parameters if specified
   - ``id``: Unique integer identifier
   - ``bad``: Boolean flag for problematic spectra

2. **grid_parameters** table: Metadata about the grid structure

Notes
-----

- The command will overwrite existing database files without warning
- All FITS files must contain the specified header keywords, or the command will fail
- The database stores relative paths, so the ``--prefix`` must be used consistently in subsequent commands
- Templates are assigned sequential integer IDs starting from 0

Error Handling
--------------

Common errors and solutions:

**"No FITS templates found"**
    Check that the ``--prefix`` path is correct and contains FITS files matching the ``--glob_mask`` pattern.

**"Keyword for [parameter] [keyword] not found"**
    The specified FITS header keyword doesn't exist in one or more template files. Check the FITS headers or adjust the keyword parameters.

See Also
--------

- :doc:`rvs_make_interpol` - Next step: create interpolated spectra
- :doc:`rvs_make_nd` - Create n-dimensional interpolation
- :doc:`rvs_make_ccf` - Create cross-correlation functions
RVSpecFit: Automated Spectroscopic Pipeline
============================================

RVSpecFit is an automated spectroscopic pipeline for determining radial velocities and stellar atmospheric parameters from spectroscopic data. It uses interpolated synthetic spectral templates (primarily PHOENIX) and performs maximum likelihood fitting.

.. image:: https://github.com/segasai/rvspecfit/actions/workflows/test.yml/badge.svg
   :target: https://github.com/segasai/rvspecfit/actions/workflows/test.yml
   :alt: Build Status

.. image:: https://readthedocs.org/projects/rvspecfit/badge/?version=latest
   :target: https://rvspecfit.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/segasai/rvspecfit/badge.svg?branch=master
   :target: https://coveralls.io/github/segasai/rvspecfit?branch=master
   :alt: Coverage Status

**Author:** Sergey Koposov (skoposov@ed.ac.uk), University of Edinburgh

**Citation:** If you use RVSpecFit, please cite:
- `Koposov et al. 2019 (ASCL) <https://ui.adsabs.harvard.edu/abs/2019ascl.soft07013K/abstract>`_
- `Koposov & Bartunov 2011 (ApJ) <https://ui.adsabs.harvard.edu/abs/2011ApJ...736..146K/abstract>`_

Quick Start
-----------

Installation
~~~~~~~~~~~~

Install the latest version directly from GitHub:

.. code-block:: bash

   pip install https://github.com/segasai/rvspecfit/archive/master.zip

Basic Usage
~~~~~~~~~~~

Here's a minimal example of fitting a spectrum:

.. code-block:: python

   from rvspecfit import fitter_ccf, vel_fit, spec_fit, utils
   import astropy.table as atpy
   
   # Load configuration (optional)
   config = utils.read_config('config.yaml')
   
   # Load your spectroscopic data
   tab = atpy.Table().read('spec.fits')
   wavelength = tab['wavelength']  # in Angstroms
   spec = tab['spec']             # flux
   espec = tab['espec']           # error spectrum
   
   # Create SpecData object
   specdata = [spec_fit.SpecData('mysetup', wavelength, spec, espec)]
   
   # Get initial parameter guess using cross-correlation
   res = fitter_ccf.fit(specdata, config)
   paramDict0 = res['best_par']
   if res['best_vsini'] is not None:
       paramDict0['vsini'] = res['best_vsini']
   
   # Perform maximum likelihood fitting
   options = {'npoly': 10}
   res1 = vel_fit.process(specdata, paramDict0, 
                         fixParam=[], config=config, options=options)
   print(res1)

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   tutorial
   api_reference
   template_preparation
   advanced_usage

.. toctree::
   :maxdepth: 2
   :caption: Configuration Commands:
   
   rvs_read_grid
   rvs_make_interpol
   rvs_make_nd
   rvs_make_ccf

.. toctree::
   :maxdepth: 2
   :caption: Analysis Commands:
   
   rvs_desi_fit

Installation and Setup
======================

Dependencies
~~~~~~~~~~~~

Required packages:
- numpy
- scipy  
- astropy
- pyyaml
- matplotlib
- numdifftools
- pandas

Optional packages:
- torch (for neural network interpolation)
- scikit-learn

PHOENIX Library Setup
~~~~~~~~~~~~~~~~~~~~~

RVSpecFit requires the PHOENIX spectral library v2.0. Download it from:
https://phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/

.. code-block:: bash

   wget -r -np -l 10 https://phoenix.astro.physik.uni-goettingen.de/data/v2.0/HiResFITS/

Template Preparation
~~~~~~~~~~~~~~~~~~~~

Before using RVSpecFit, you need to prepare template files for your spectral configuration using the four-step pipeline:

1. **Create PHOENIX database** (see :doc:`rvs_read_grid` for full details):

.. code-block:: bash

   rvs_read_grid --prefix /path/to/PHOENIX/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/ --templdb files.db

2. **Make interpolated spectra** (see :doc:`rvs_make_interpol` for full details):

.. code-block:: bash

   rvs_make_interpol --setup mysetup --lambda0 4000 --lambda1 5000 \
       --resol_func '1000+2*x' --step 0.5 --templdb files.db \
       --oprefix /template/path/ --templprefix /PHOENIX/path/ \
       --wavefile WAVE_PHOENIX-ACES-AGSS-COND-2011.fits --air --revision=v2020x

3. **Create n-d interpolator** (see :doc:`rvs_make_nd` for full details):

.. code-block:: bash

   rvs_make_nd --prefix /template/path/ --setup mysetup --revision=v2020x

4. **Make cross-correlation files** (see :doc:`rvs_make_ccf` for full details):

.. code-block:: bash

   rvs_make_ccf --setup mysetup --lambda0 4000 --lambda1 5000 \
       --every 30 --vsinis 0,10,300 --prefix /template/path/ \
       --step 0.5 --revision=v2020x

5. **Create configuration file (config.yaml):**

.. code-block:: yaml

   template_lib: '/path/to/template_data/'
   min_vel: -1500
   max_vel: 1500
   min_vel_step: 0.2
   vel_step0: 5
   min_vsini: 0.01
   max_vsini: 500
   second_minimizer: 1

API Reference
=============

Core Functions
~~~~~~~~~~~~~~

These are the main functions users interact with:

Cross-Correlation Fitting
--------------------------

.. autofunction:: rvspecfit.fitter_ccf.fit

Maximum Likelihood Fitting  
---------------------------

.. autofunction:: rvspecfit.vel_fit.process

.. autofunction:: rvspecfit.vel_fit.firstguess

Core Data Structures
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rvspecfit.spec_fit.SpecData
   :members:
   :show-inheritance:

Spectral Likelihood
~~~~~~~~~~~~~~~~~~~

.. autofunction:: rvspecfit.spec_fit.get_chisq

Configuration Utilities
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rvspecfit.utils.read_config

.. autofunction:: rvspecfit.utils.get_default_config

Advanced Usage
==============

Multiple Spectra Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~

RVSpecFit can simultaneously fit multiple spectra from different instrument arms or exposures:

.. code-block:: python

   # Multiple instrument arms
   sd_blue = spec_fit.SpecData('blue', wavelength1, spec1, espec1, badmask=badmask1)
   sd_red = spec_fit.SpecData('red', wavelength2, spec2, espec2, badmask=badmask2)
   
   res = vel_fit.process([sd_blue, sd_red], paramDict0, 
                        fixParam=[], config=config, options=options)

Likelihood Function Access
~~~~~~~~~~~~~~~~~~~~~~~~~~

For integration with larger inference frameworks:

.. code-block:: python

   vel = 300  # km/s
   atm_params = [4100, 4, -2.5, 0.6]  # teff, logg, feh, alpha
   
   spec_chisq = spec_fit.get_chisq(specdata, vel, atm_params, None,
                                  config=config, options={'npoly': 10})
   loglike = -0.5 * spec_chisq

DESI/WEAVE Integration
~~~~~~~~~~~~~~~~~~~~~~

For DESI and WEAVE data, use the specialized fitting routines:

.. code-block:: bash

   rvs_desi_fit input_spectrum.fits
   rvs_weave_fit input_spectrum.fits

Template Preparation Pipeline
===============================

RVSpecFit provides a four-step command-line pipeline for preparing stellar template libraries. Each step must be executed in order:

**Step 1:** :doc:`rvs_read_grid` - Create SQLite database from FITS template files

**Step 2:** :doc:`rvs_make_interpol` - Process and interpolate template spectra to target resolution

**Step 3:** :doc:`rvs_make_nd` - Create n-dimensional interpolation structures

**Step 4:** :doc:`rvs_make_ccf` - Generate FFT-based cross-correlation templates

See the individual command documentation for detailed usage instructions and examples.

Template Preparation Tools
===========================

Command-Line Tools
~~~~~~~~~~~~~~~~~~

RVSpecFit includes several command-line tools for template preparation:

Database Creation
-----------------

.. automodule:: rvspecfit.read_grid
   :members: main
   :noindex:

Interpolation
-------------

.. automodule:: rvspecfit.make_interpol
   :members: main
   :noindex:

N-D Interpolator
----------------

.. automodule:: rvspecfit.make_nd
   :members: main
   :noindex:

Cross-Correlation Templates
---------------------------

.. automodule:: rvspecfit.make_ccf
   :members: main
   :noindex:

Interpolation Methods
=====================

RVSpecFit supports multiple interpolation methods:

1. **Multi-dimensional linear interpolation** (default) - Uses Delaunay triangulation
2. **Multilinear interpolation** - For regular n-D grids without gaps  
3. **Neural Network interpolation** - Alternative method using PyTorch

Neural Network Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train a neural network interpolator instead of using linear interpolation:

.. code-block:: bash

   rvs_train_nn_interpolator --dir /template/path/ --setup mysetup

Set the environment variable to force CPU usage:

.. code-block:: bash

   export RVS_NN_DEVICE=cpu

Custom Template Libraries
==========================

While RVSpecFit is designed for PHOENIX, you can use custom synthetic spectral libraries. Requirements:

- Wavelength FITS file
- Collection of FITS files with spectra and stellar parameters in headers (LOGG, TEFF, etc.)
- Use ``rvs_read_grid`` to create database, then follow standard preparation steps

Spectral Configurations
========================

A spectral configuration defines the combination of wavelength range and resolution for your instrument setup. Examples:

- Giraffe/HR21
- 2DF/1700D  
- DESI blue arm

Each configuration requires separate template preparation but can be combined in joint fits.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex` 
* :ref:`search`

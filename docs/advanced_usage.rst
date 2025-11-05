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

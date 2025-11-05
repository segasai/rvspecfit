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
- `Koposov et al. 2011 (ApJ) <https://ui.adsabs.harvard.edu/abs/2011ApJ...736..146K/abstract>`_

Quick Start
-----------

Installation
~~~~~~~~~~~~

Install the latest version from PyPI:

.. code-block:: bash

   pip install rvspecfit

Basic Usage
~~~~~~~~~~~

Here's a minimal example of fitting a spectrum (but keep in mind that you will need to build the library of templates first).

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

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex` 
* :ref:`search`

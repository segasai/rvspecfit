Template Preparation
====================

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

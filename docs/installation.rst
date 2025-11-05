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

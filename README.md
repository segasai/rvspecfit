[![Build Status](https://travis-ci.org/segasai/rvspecfit.svg?branch=master)](https://travis-ci.org/segasai/rvspecfit)

Automated spectroscopic pipeline to determine radial velocities and 
stellar atmospheric parameters
Author: Sergey Koposov skoposov@cmu.edu, Carnegie Mellon University

Dependencies: 
numpy, scipy, astropy, pyyaml, frozendict

##  Running on DESI/WEAVE data

To run on DESI data use rvs_desi_fit or rvs_weave_fit

## Creation of the template grid library 
Currently only PHOENIX library is supported. 

The first step is to read the template grid into the sqlite database

rvs_read_grid --prefix PATH_TO_PHOENIX/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/  --templdb ../templ_data/files.db

Then we create the interpolated spectra for different instrument setups
rvs_make_interpol --setup test_setup --lambda0 4000 --lambda1 5000 --resol 5000 --step 0.5 --templdb ../templ_data/files.db --oprefix ../templ_data/ --templprefix /home/skoposov/science/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/ --wavefile /home/skoposov/science/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits

Create the triangulation
rvs_make_nd  --prefix ../templ_data/ --setup test_setup

Create the Fourier transformation of the templates
rvs_make_ccf --setup test_setup --lambda0 4000 --lambda1 5000 --step 0.5 --prefix ../templ_data/ --vsinis 0,300 --every 50 --oprefix=../templ_data

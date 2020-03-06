[![Build Status](https://travis-ci.org/segasai/rvspecfit.svg?branch=master)](https://travis-ci.org/segasai/rvspecfit)

Automated spectroscopic pipeline to determine radial velocities and 
stellar atmospheric parameters
Author: Sergey Koposov skoposov@cmu.edu, Carnegie Mellon University

Dependencies: 
numpy, scipy, astropy, pyyaml, matplotlib, numdifftools, pandas

##  Running on DESI/WEAVE data

To run on DESI data use rvs_desi_fit or rvs_weave_fit

## Creation of the template grid library 
Currently only PHOENIX library is supported. 

- The first step is to read the template grid into the sqlite database

`rvs_read_grid --prefix PATH_TO_PHOENIX/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/  --templdb files.db`

If you have a different set of templates from PHOENIX. You may need to populate that database yourself.

- Then we create the interpolated spectra for different instrument configurations

`rvs_make_interpol --setup test_setup --lambda0 4000 --lambda1 5000 --resol 5000 --step 0.5 --templdb ../templ_data/files.db --oprefix ../templ_data/ --templprefix /home/skoposov/science/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/ --wavefile /home/skoposov/science/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits`
For example this command will create a grid for the instrument configuration with the the wavelength range from 4000 to 5000, the pixel step of 0.5 angstrom and resolution of 5000. The files will then be placed in the directory specified by oprefix

- Create the triangulation
`rvs_make_nd  --prefix ../templ_data/ --setup test_setup`

- Create the Fourier transformations of template for the cross-correlation steps
rvs_make_ccf --setup test_setup --lambda0 4000 --lambda1 5000 --step 0.5 --prefix ../templ_data/ --vsinis 0,300 --every 50 --oprefix=../templ_data


Now to actually use the code 

You need just a few commands

    ```
from rvspecfit import fitter_ccf, vel_fit, spec_fit, utils
config=utils.read_config('config.yaml') # optional

# This constructs the specData object
sd = spec_fit.SpecData('test_setup',
                               wavelength,
                               spec,
                               espec,
                               badmask=badmask)    

# this tries to get a sensible guess for the stellar parameters/velocity
res = fitter_ccf.fit(specdata, config)
paramDict0 = res['best_par']
fixParam = [] 
if res['best_vsini'] is not None:
    paramDict0['vsini'] = res['best_vsini']

options = {'npoly':10}
# this does the actual fitting
res1 = vel_fit.process(specdata,
                           paramDict0,
                           fixParam=fixParam,
                           config=config,
                           options=options)
print(res1)

```
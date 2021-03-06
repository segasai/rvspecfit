[![Build Status](https://travis-ci.org/segasai/rvspecfit.svg?branch=master)](https://travis-ci.org/segasai/rvspecfit)

Automated spectroscopic pipeline to determine radial velocities and 
stellar atmospheric parameters
Author: Sergey Koposov skoposov _AT_ ed _DOT_ ac _DOT uk, 
University of Edinburgh

Dependencies: 
numpy, scipy, astropy, pyyaml, matplotlib, numdifftools, pandas

# Introduction

The code here can perform radial velocity fitting and general spectral fitting 
with templated to any spectroscopic data.
To get started you will need to

* Install rvspecfit 
* Download PHOENIX library 
* Create various PHOENIX processed files (such as spectra convolved to the 
resolution of your instrument, interpolation files)
* Run the code 


## Installation

To install you can just do can just do 
`
 pip install https://github.com/segasai/rvspecfit/archive/master.zip
`
which will install the latest version


## PHOENIX library

The library is avialable here  ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/


## Preparation of PHOENIX library

The preparation requires several steps

* Creating a PHOENIX file sqlite database that will be used in the processing.

This is done with 
```
$ rvs_read_grid --prefix $PATH/PHOENIX/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/ --templdb files.db
```

* Making interpolated spectra
```
$ rvs_make_interpol --setup myconf --lambda0 4000 --lambda1 5000 \
    --resol_func '1000+2000*x' --step 0.5 --templdb ${PREFIX}/files.db \
    --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $PATH/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits \
    --air --revision=v2020x
```
That will create the spectral configuration called myconf for spectra with wavelength range of 4000 to 5000, step 0.5 angstrom and resolution being 1000+2000*wavelength. It also requires paths to the files.db database created at previous step as well as the file with the wavelength grid of PHOENIX library which is distributed with PHOENIX. You can also choose to do things in air or vacuum. 
This step will take up to an hour. 


* Making the n-d interpolator. 
That requires the path to the files created at previous step. 
```
$ rvs_make_nd --prefix ${PREFIX}/ --setup myconf --revision=v2020x
```

* Making the cross-correlation files

```
$ rvs_make_ccf --setup myconf --lambda0 4000 --lambda1 5000  \
    --every 30 --vsinis 0,10,300 --prefix ${PREFIX}/ --oprefix=${PREFIX} \
    --step 0.5 --revision=v2020x
```

That creates a list of Fourier transformed templates for the CCF. IN this 
case this list will have every 30-th spectra from the database. It also uses a 
list of Vsini's  of  0,10,300 when creating CCF templates.


After that you should be able to use rvspecfit. 

## Code example

You need just a few commands

```python
from rvspecfit import fitter_ccf, vel_fit, spec_fit, utils
config=utils.read_config('config.yaml') # optional
# you can create a configuration file with various options


# let's assume we have data stored in 1d arrays in a table
tab=atpy.Table().read('spec.fits')
wavelength = tab['wavelength']
spec = tab['spec']
espec = tab['espec']
badmask = tab['badmask']


# This constructs the specData object from wavelength, spectrum and error
spectrum arrays .
The rvspecfit works on arrays of SpecData's

sd = spec_fit.SpecData('myconf',
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

# this does the actual fitting performing the maximum like-lihood fitting of the data
res1 = vel_fit.process(specdata,
                           paramDict0,
                           fixParam=fixParam,
                           config=config,
                           options=options)
print(res1)

```

##  Likelihood function

One advantage is that you can use rvspecfit as a part of larger inference framework. I.e. you can add the 
spectral likelihood to other likelihood terms. 

To do that you just need this call

```python
vel=300
atm_params = [4000,4, -2.5, 0.6] # 'teff', 'logg', 'feh', 'alpha'
spec_chisq  = spec_fit.get_chisq(specdata,
                             vel,
                             atm_params,
                             None,
                             None,
                             config=config, options = dict(npoly=10))
```                            
That will compute the chi-square of the spectrum for a given template and radial velocity. That will also use
the 10 order polynomial as multiplicative continuum . 
The resulting chi-square can be then multiplied by (-0.5) and added to your log-likelihood if needed.


##  Running on DESI/WEAVE data

To run on DESI data use rvs_desi_fit or rvs_weave_fit


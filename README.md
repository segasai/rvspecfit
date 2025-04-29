![Build Status](https://github.com/segasai/rvspecfit/actions/workflows/test.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/rvspecfit/badge/?version=latest)](https://rvspecfit.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/segasai/rvspecfit/badge.svg?branch=master)](https://coveralls.io/github/segasai/rvspecfit?branch=master)

Automated spectroscopic pipeline to determine radial velocities and 
stellar atmospheric parameters

Author: Sergey Koposov skoposov _AT_ ed _DOT_ ac _DOT uk, 
University of Edinburgh

Citation: If you use it, please cite
https://ui.adsabs.harvard.edu/abs/2019ascl.soft07013K/abstract
and https://ui.adsabs.harvard.edu/abs/2011ApJ...736..146K/abstract

If something doesn't quite work, I'm happy to help

Dependencies: 
numpy, scipy, astropy, pyyaml, matplotlib, numdifftools, pandas

Optional: torch, scikit-learn


# Introduction

The code here can perform radial velocity fitting and general spectral fitting
with interpolated templates to any spectroscopic data.

The library is primarily used with the PHOENIX library, but you can also use
your own library of synthetic spectra (see below)

To get started you will need to

* Install rvspecfit 
* Download PHOENIX library 
* Create PHOENIX processed files for your spectral configuration. This involves convolving spectra to
resolution of your instrument and producing interpolators.
* Create a configuration file 
* Run the code


## Installation

To install you can just do can just do 
`
 pip install https://github.com/segasai/rvspecfit/archive/master.zip
`

which will install the latest version


## PHOENIX library

The library is avialable here https://phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/
(Make sure you get the v2.0 one!). If you want to download the whole thing you will likely need a command like
`
wget -r -np -l 10 https://phoenix.astro.physik.uni-goettingen.de/data/v2.0/HiResFITS/
`

## Spectral configuration

In the following text we use the idea of spectral configuration. I.e. this is the combination wavelength range and resolution. rvspecfit is designed to construct interpolated spectra for different spectral configurations, model them separately and together. Typically a spectral configuration would be instrument + arm . I.e. Giraffe/HR21, or 2DF/1700D or DESI blue arm.

## Preparation of PHOENIX templates

The preparation requires several steps (you can find several examples of these steps in the surveys folder, i.e surveys/sdss/make_sdss.sh surveys/desi/make_desi.sh which prepare rvspecfit for processing SDSS or DESI spectra).

The commands below all support --help option, so you can always run that to get more help on the meaning of different options.

* Creating a PHOENIX file sqlite database that will be used in the processing (you only need to do this step once)

This is done with
```
$ rvs_read_grid --prefix $PATH/PHOENIX/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/ --templdb files.db
```
Some templates may be marked as bad there if you wish.

* Making interpolated spectra for your spectral configuration.
If your instrument has multiple arms (i.e. blur/red), you will need to run this once for each arm.

```
$ rvs_make_interpol --setup mysetup --lambda0 4000 --lambda1 5000 \
    --resol_func '1000+2*x' --step 0.5 --templdb ${PREFIX}/files.db \
    --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $PATH/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits \
    --air --revision=v2020x
```

That will create the spectral configuration called mysetup for spectra with wavelength range of 4000 to 5000, step 0.5 Angstrom and resolution being 1000+2*x (where x is wavelength in Angstroms). It also requires paths to the files.db database created at previous step as well as the file with the wavelength grid of PHOENIX library which is distributed with PHOENIX. You can also choose to do things in air or vacuum. TEMPLPREF is the path to the PHOENIX library.

This step can take up to an hour, depending on your machine.

* Making the n-d interpolator.
That requires the path to the files created at previous step.
```
$ rvs_make_nd --prefix ${PREFIX}/ --setup mysetup --revision=v2020x
```
This will create either a polylinear interpolator or Linear interpolator within Voronoi cells.

* Neural network interpolator
An alternative to polylinear interpolator is a neural network based interpolator.
This requires running
```
$ rvs_train_nn_interpolator  --dir ${PREFIX}/ --setup mysetup
```
instead of rvs_make_nd. 
When doing the fitting, you may also set this
environmental variable RVS_NN_DEVICE=cpu to force the code to run on cpu.

* Making the cross-correlation files

```
$ rvs_make_ccf --setup mysetup --lambda0 4000 --lambda1 5000  \
    --every 30 --vsinis 0,10,300 --prefix ${PREFIX}/ --oprefix=${PREFIX} \
    --step 0.5 --revision=v2020x
```

That creates a list of Fourier transformed templates for the CCF. IN this 
case this list will have every 30-th spectra from the database. It also uses a 
list of Vsini's  of  0,10,300 when creating CCF templates.

* The very last step is creating of the configuration file config.yaml with contents like this
```yaml
template_lib : '/home/username/template_data/'
min_vel: -1500
max_vel: 1500
min_vel_step: 0.2
vel_step0: 5
min_vsini: 0.01
max_vsini: 500
second_minimizer: 1
```
where the only important parameter is the template_lib path which points at the location where all
the processed template files are.

After that you should be able to use rvspecfit. 




## Code example

You need just a few commands

```python
from rvspecfit import fitter_ccf, vel_fit, spec_fit, utils
config=utils.read_config('config.yaml') # optional
# you can create a configuration file with various options


# let's assume we have data stored in 1d arrays in a table
# with wavelength being in Angstrom
# espec being a vector of standard deviations

tab=atpy.Table().read('spec.fits')
wavelength = tab['wavelength']
spec = tab['spec']
espec = tab['espec']

# This constructs the specData object from wavelength, spectrum and error
# spectrum arrays. The rvspecfit works on arrays of SpecData's which may
# represent multiple exposures or multiple spectral configurations
# Here we just have one spectrum from the spectral configuration "mysetup"

specdata = [spec_fit.SpecData('mysetup',
                               wavelength,
                               spec,
                               espec)
                              ]


# this tries to get a sensible guess for the stellar parameters/velocity
res = fitter_ccf.fit(specdata, config)
paramDict0 = res['best_par']
fixParam = [] 
if res['best_vsini'] is not None:
    paramDict0['vsini'] = res['best_vsini']

options = {'npoly':10}

# this does the actual fitting performing the maximum likelihood fitting of the data
res1 = vel_fit.process(specdata,
                           paramDict0,
                           fixParam=fixParam,
                           config=config,
                           options=options)
print(res1)

```
## Fitting multiple spectra simultaneously 

If your have multiple spectra, from multiple instrument arms, from multiple 
exposures, fitting them is easy. 

If you have multiple instrument arms, you
will need to prepare the spectral interpolators for all the arms 
and then combine the specdata objects from multiple arms in the list

```python
sd_blue = spec_fit.SpecData('blue',
                               wavelength1,
                               spec1,
                               espec1,
                               badmask=badmask1)
sd_red = spec_fit.SpecData('red',
                               wavelength2,
                               spec2,
                               espec2,
                               badmask=badmask2)
res1 = vel_fit.process([sd_blue, sd_red],
                           paramDict0,
                           fixParam=fixParam,
                           config=config,
                           options=options)
```
If you just have multiple exposures, you can obviously use the 
same spectral configuration for them and then combine the specdata objects for
those

##  Likelihood function

One advantage of rvspecfit is that you can use it as  part of larger inference framework. 
I.e. you can add the  spectral likelihood to other likelihood terms. 

To do that you just need this call the get_chisq() function that will 
return the -2*log(L) (or chi-square) of your spectral dataset given the 
model parameters

```python
loglike = 0 
vel=300
atm_params = [4100,4, -2.5, 0.6] # 'teff', 'logg', 'feh', 'alpha'
spec_chisq  = spec_fit.get_chisq(specdata,
                             vel,
                             atm_params,
                             None,
                             config=config, options = dict(npoly=10))
loglike += -0.5 * spec_chisq
```                            
That will compute the chi-square of the spectrum for a given template parameters
and radial velocity. That will also use
the 10 order polynomial as multiplicative continuum . 
The resulting chi-square can be then multiplied by (-0.5) and added to your log-likelihood if needed.


##  Running on DESI/WEAVE data

To run on DESI data use rvs_desi_fit or rvs_weave_fit

## Interpolation methods 

By default rvspecfit uses the multi-dimensional linear interpolation with Delaunay 
triangulation. It also supports the multilinear interpolation if 
the input grid is organized as a true n-D grid without gaps.
The alternative interpolation method is Neural Network based.

## Other template libraries

You can substitute PHOENIX by your own library of synthetic spectra. The only thing needed there is a wavelength FITS file and a collection of FITS files with spectra and stellar parameter information in the headers (i.e. LOGG, TEFF keywords etc). You can then use rvs_read_grid to create a database of spectra and then follow the same steps as with PHOENIX

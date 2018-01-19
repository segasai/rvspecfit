#!/bin/bash -e 
PREFIX=../../templ_data
TEMPLPREF=/home/skoposov/science/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/
WAVEFILE=/home/skoposov/science/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
RLAM0=5700
RLAM1=10000
BLAM0=3600
BLAM1=6200
VSINIS=0,300
RESOL=5000

rvs_make_interpol --air --setup weave_r --lambda0 $RLAM0 --lambda1 $RLAM1 --resol 5000 --step 0.25 --templdb ${PREFIX}/files.db --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $WAVEFILE

rvs_make_nd --prefix ${PREFIX}/ --setup weave_r

rvs_make_ccf --setup weave_r --lambda0 $RLAM0 --lambda1 $RLAM1 --every 30 --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX} --step 0.25


rvs_make_interpol --air --setup weave_b --lambda0 $BLAM0 --lambda1 $BLAM1 --resol 5000 --step 0.25 --templdb ${PREFIX}/files.db --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $WAVEFILE

rvs_make_nd --prefix ${PREFIX}/ --setup weave_b

rvs_make_ccf --setup weave_b --lambda0 $BLAM0 --lambda1 $BLAM1 --every 30 --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX} --step 0.25


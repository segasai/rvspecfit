#!/bin/bash -e 
PREFIX=./templ_data
TEMPLPREF=small_phoenix/
WAVEFILE=small_phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
BLAM0=3500
BLAM1=5900
VSINIS=0,300
RNAME=aat1_1700d
BNAME=aat1_580v
BSTEP=0.5
BRESOL=1450

rvs_read_grid --prefix $TEMPLPREF --templdb $PREFIX/files.db

rvs_make_interpol --air --setup $BNAME --lambda0 $BLAM0 --lambda1 $BLAM1 --resol $BRESOL --step $BSTEP --templdb ${PREFIX}/files.db --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --fixed_fwhm --wavefile $WAVEFILE

rvs_make_nd --prefix ${PREFIX}/ --setup $BNAME

rvs_make_ccf --setup $BNAME --lambda0 $BLAM0 --lambda1 $BLAM1  --every 3 --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX} --step $BSTEP

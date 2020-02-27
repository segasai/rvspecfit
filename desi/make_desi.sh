#!/bin/bash -e 
PREFIX=../../templ_data
TEMPLPREF=/home/skoposov/science/PHOENIX/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/
WAVEFILE=/home/skoposov/science/PHOENIX/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
RLAM0=5500
RLAM1=7900
BLAM0=3300
BLAM1=6100
ZLAM0=7300
ZLAM1=10000
BRESOL=2800
RRESOL=3800
ZRESOL=5500
BRZRESOL=3800
BRZLAM0=3300
BRZLAM1=10000
VSINIS=0,300

rvs_make_interpol --setup desi_b --lambda0 $BLAM0 --lambda1 $BLAM1 \
    --resol $BRESOL --step 0.4 --templdb ${PREFIX}/files.db \
    --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $WAVEFILE \
    --fixed_fwhm


rvs_make_nd --prefix ${PREFIX}/ --setup desi_b

rvs_make_ccf --setup desi_b --lambda0 $BLAM0 --lambda1 $BLAM1  --every 30 \
    --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX} --step 0.4

rvs_make_interpol --setup desi_r --lambda0 $RLAM0 --lambda1 $RLAM1 \
    --resol $RRESOL --step 0.5 --templdb ${PREFIX}/files.db \
    --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $WAVEFILE \
    --fixed_fwhm

rvs_make_nd --prefix ${PREFIX}/ --setup desi_r

rvs_make_ccf --setup desi_r --lambda0 $RLAM0 --lambda1 $RLAM1 --every 30 \
    --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX} --step 0.5


rvs_make_interpol --setup desi_z --lambda0 $ZLAM0 --lambda1 $ZLAM1 \
    --resol $ZRESOL --step 0.5 --templdb ${PREFIX}/files.db \
    --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $WAVEFILE \
    --fixed_fwhm

rvs_make_nd --prefix ${PREFIX}/ --setup desi_z

rvs_make_ccf --setup desi_z --lambda0 $ZLAM0 --lambda1 $ZLAM1  --every 30 \
    --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX} --step 0.5


rvs_make_interpol --setup desi_brz --lambda0 $BRZLAM0 --lambda1 $BRZLAM1 \
    --resol $BRZRESOL --step 0.5 --templdb ${PREFIX}/files.db \
    --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $WAVEFILE \
    --fixed_fwhm

rvs_make_nd --prefix ${PREFIX}/ --setup desi_brz

rvs_make_ccf --setup desi_brz --lambda0 $BRZLAM0 --lambda1 $BRZLAM1 \
    --every 30 --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX} \
    --step 0.5

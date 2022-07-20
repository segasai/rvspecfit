#!/bin/bash
set -e
TEMPLPREF=/home/skoposov/science/PHOENIX/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/
WAVEFILE=/home/skoposov/science/PHOENIX/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
LAM0=8410 # 8460- 50
LAM1=8750 # 8750 + 50
STEP=0.05
RESOL=11500
VSINIS=0,300
REVISION=v220719
EVERY=200
SMOOTH=0.0
PREFIX=/home/skoposov/science/specfit/templ_data/gaiarvs/${REVISION}/

rvs_make_interpol --nthreads 2 --setup gaiarvs --lambda0 $LAM0 --lambda1 $LAM1 \
    --resol_func $RESOL --step $STEP --templdb ${PREFIX}/files.db \
    --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $WAVEFILE \
    --revision=$REVISION --no-normalize

rvs_regularize_grid --input ${PREFIX}/specs_gaiarvs.pkl --smooth $SMOOTH --output ${PREFIX}/specs_gaiarvs.pkl

rvs_make_nd --regular --prefix ${PREFIX}/ --setup gaiarvs --revision=$REVISION

rvs_make_ccf --setup gaiarvs --lambda0 $LAM0 --lambda1 $LAM1  --every $EVERY \
    --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX} --step $STEP \
    --revision=$REVISION

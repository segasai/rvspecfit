#!/bin/bash
set -e
PREFIX=../../templ_data
TEMPLPREF=/home/skoposov/science/PHOENIX/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/
WAVEFILE=/home/skoposov/science/PHOENIX/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
BLAM0=3500
BLAM1=5900
RLAM0=5660
RLAM1=7720
ZLAM0=7420
ZLAM1=9924
BSTEP=0.4
RSTEP=0.4
ZSTEP=0.4
BRZSTEP=0.4
BRESOL='x/1.55'
RRESOL='x/1.55'
ZRESOL='x/1.8'
BRZLAM0=$BLAM0
BRZLAM1=$ZLAM1
VSINIS=0,300
REVISION=v200304

# B

rvs_make_interpol --setup desi_b --lambda0 $BLAM0 --lambda1 $BLAM1 \
    --resol_func $BRESOL --step $BSTEP --templdb ${PREFIX}/files.db \
    --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $WAVEFILE \
    --revision=$REVISION --no-normalize

rvs_make_nd --prefix ${PREFIX}/ --setup desi_b --revision=$REVISION

rvs_make_ccf --setup desi_b --lambda0 $BLAM0 --lambda1 $BLAM1  --every 30 \
    --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX} --step $BSTEP \
    --revision=$REVISION

# R

rvs_make_interpol --setup desi_r --lambda0 $RLAM0 --lambda1 $RLAM1 \
    --resol_func $RRESOL --step $RSTEP --templdb ${PREFIX}/files.db \
    --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $WAVEFILE \
    --revision=$REVISION --no-normalize

rvs_make_nd --prefix ${PREFIX}/ --setup desi_r --revision=$REVISION

rvs_make_ccf --setup desi_r --lambda0 $RLAM0 --lambda1 $RLAM1 --every 30 \
    --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX} --step $RSTEP \
    --revision=$REVISION

# Z

rvs_make_interpol --setup desi_z --lambda0 $ZLAM0 --lambda1 $ZLAM1 \
    --resol_func $ZRESOL --step $ZSTEP --templdb ${PREFIX}/files.db \
    --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $WAVEFILE \
    --revision=$REVISION --no-normalize

rvs_make_nd --prefix ${PREFIX}/ --setup desi_z --revision=$REVISION

rvs_make_ccf --setup desi_z --lambda0 $ZLAM0 --lambda1 $ZLAM1  --every 30 \
    --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX} --step $ZSTEP \
    --revision=$REVISION


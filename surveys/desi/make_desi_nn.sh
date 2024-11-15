#!/bin/bash
set -e
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
BRESOL='x/1.55'
RRESOL='x/1.55'
ZRESOL='x/1.8'
VSINIS=0,300
REVISION=v241115_phoenn
EVERY=200
SMOOTH=0.0
PREFIX=../../..//templ_data/desi/${REVISION}/

DBFILE0=${PREFIX}/files.db
DBFILE=${PREFIX}/files_masked.db

mkdir -p $PREFIX

rvs_read_grid --prefix /home/skoposov/science/PHOENIX/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/ --templdb $DBFILE0
../mask_phoenix_grid.sh $DBFILE0 $DBFILE


LAM0=($BLAM0 $RLAM0 $ZLAM0)
LAM1=($BLAM1 $RLAM1 $ZLAM1)
STEP=($BSTEP $RSTEP $ZSTEP)
CONF=(desi_b desi_r desi_z)
RESOL=($BRESOL $RRESOL $ZRESOL)

for i in 0 1 2; do {
    (
    CURLAM0=${LAM0[$i]}
    CURLAM1=${LAM1[$i]}
    CURCONF=${CONF[$i]}
    CURSTEP=${STEP[$i]}
    CURRESOL=${RESOL[$i]}
    rvs_make_interpol --nthreads 2 --setup $CURCONF --lambda0 $CURLAM0 --lambda1 $CURLAM1 \
    --resol_func $CURRESOL --step $CURSTEP --templdb ${DBFILE} \
    --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $WAVEFILE \
    --revision=$REVISION --no-normalize

    rvs_make_ccf --setup $CURCONF --lambda0 $CURLAM0 --lambda1 $CURLAM1  --every $EVERY \
    --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX} --step $CURSTEP \
    --revision=$REVISION

    rvs_make_ccf --setup $CURCONF --lambda0 $CURLAM0 --lambda1 $CURLAM1  --every $EVERY \
    --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX} --step $CURSTEP \
    --revision=$REVISION --nocontinuum
    ) & 
} ; done 
wait;

for a in b r z ; do rvs_train_nn_interpolator  --pca_init --revision=$REVISION   --dir=/home/skoposov/science/specfit/templ_data/desi/${REVISION}/  --setup=desi_${a} ; done

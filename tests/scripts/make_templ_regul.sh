#!/bin/bash -e
DIR=`dirname $0`/../
PREFIX=$DIR//templ_data
TEMPLPREF=$DIR/small_phoenix/
WAVEFILE=$DIR/small_phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
BLAM0=3500
BLAM1=5900
VSINIS=0,300
BNAME=aat_reg
BSTEP=0.5
BRESOL=1450
COV="echo"

RVS_READ_GRID="$COV `command -v rvs_read_grid`"
RVS_MAKE_INTERPOL="$COV `command -v rvs_make_interpol`"
RVS_REGULARIZE_GRID="$COV `command -v rvs_regularize_grid`"


$RVS_READ_GRID --prefix $TEMPLPREF --templdb $PREFIX/files.db
$RVS_MAKE_INTERPOL --air --setup $BNAME --lambda0 $BLAM0 --lambda1 $BLAM1 --resol $BRESOL --step $BSTEP --templdb ${PREFIX}/files.db --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --fixed_fwhm --wavefile $WAVEFILE --nthreads 1

$RVS_REGULARIZE_GRID --input $PREFIX/specs_$BNAME.pkl --output $PREFIX/specs_$BNAME.pkl


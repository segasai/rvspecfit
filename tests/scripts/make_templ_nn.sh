#!/bin/bash -e
DIR=`dirname $0`/../
PREFIX0=$DIR//templ_data
PREFIX=$DIR//templ_data_nn/
TEMPLPREF=$DIR/small_phoenix/
WAVEFILE=$DIR/small_phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
BLAM0=3500
BLAM1=5900
VSINIS=0,300
RNAME=aat1_1700d
BNAME=aat1_580v
BSTEP=0.5
BRESOL=1450
COV="echo"

RVS_READ_GRID="$COV `command -v rvs_read_grid`"
RVS_MAKE_INTERPOL="$COV `command -v rvs_make_interpol`"
RVS_MAKE_ND="$COV `command -v rvs_train_nn_interpolator`"
RVS_MAKE_CCF="$COV `command -v rvs_make_ccf`"


$RVS_READ_GRID --prefix $TEMPLPREF --templdb $PREFIX0/files.db
$RVS_MAKE_INTERPOL --air --setup $BNAME --lambda0 $BLAM0 --lambda1 $BLAM1 --resol $BRESOL --step $BSTEP --templdb ${PREFIX0}/files.db --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --fixed_fwhm --wavefile $WAVEFILE --nthreads 1
$RVS_MAKE_ND --dir ${PREFIX}/ --pca_init --npc 10 --cpu --setup $BNAME

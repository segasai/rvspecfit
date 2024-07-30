#!/bin/bash -e
DIR=`dirname $0`/../
DBFILE=$DIR/files_config1_grid.db
PREF=$DIR/tmp/
PREF1=$DIR/templ_data_test/
CONF_NAME=config1_grid
STEP=1
LAM1=4550
LAM2=5450
RESOL=1000


COV="echo"

RVS_READ_GRID="$COV `command -v rvs_read_grid`"
RVS_MAKE_INTERPOL="$COV `command -v rvs_make_interpol`"
RVS_MAKE_ND="$COV `command -v rvs_make_nd`"
RVS_MAKE_CCF="$COV `command -v rvs_make_ccf`"

mkdir -p $PREF/specs
mkdir -p $PREF1
rm -f $PREF/xx*fits
python $DIR/mktemps_grid.py $PREF wave.fits
rm -f $DBFILE
$RVS_READ_GRID  --prefix $PREF --templdb $DBFILE
$RVS_MAKE_INTERPOL --templdb $DBFILE --wavefile $PREF/wave.fits --templprefix $PREF  --resol $RESOL --lambda0 $LAM1 --lambda1 $LAM2 --step $STEP --setup $CONF_NAME --oprefix $PREF1
$RVS_MAKE_ND  --setup $CONF_NAME --prefix $PREF1 --regulargrid
$RVS_MAKE_CCF --setup $CONF_NAME --prefix $PREF1 --lambda0 $LAM1 --lambda1 $LAM2 --step $STEP --every 10 --oprefix $PREF1

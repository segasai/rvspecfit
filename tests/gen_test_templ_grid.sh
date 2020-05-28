#!/bin/bash -e
DBFILE=files_config1_grid.db
PREF=tmp/
PREF1=templ_data_test/
CONF_NAME=config1_grid
STEP=1
LAM1=4550
LAM2=5450
RESOL=1000

mkdir -p $PREF/specs
mkdir -p $PREF1
rm -f $PREF/xx*fits
python mktemps.py $PREF wave.fits 300
rm -f $DBFILE
rvs_read_grid  --prefix $PREF --templdb $DBFILE
rvs_make_interpol --templdb $DBFILE --wavefile $PREF/wave.fits --templprefix $PREF  --resol $RESOL --lambda0 $LAM1 --lambda1 $LAM2 --step $STEP --setup $CONF_NAME --oprefix $PREF1
rvs_make_nd  --setup $CONF_NAME --prefix $PREF1
rvs_make_ccf  --setup $CONF_NAME --prefix $PREF1 --lambda0 $LAM1 --lambda1 $LAM2 --step $STEP --every 10 --oprefix $PREF1

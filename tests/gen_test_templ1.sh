#!/bin/bash -e
DBFILE=test.db
PREF=tmp/
PREF1=templ_data_test/
mkdir -p $PREF
mkdir -p $PREF1
rm -f $PREF/xx*fits
python mktemps.py $PREF wave.fits 3000
rm -f $DBFILE
rvs_read_grid  --prefix $PREF --templdb $DBFILE
rvs_make_interpol --templdb $DBFILE --wavefile $PREF/wave.fits --templprefix $PREF  --resol 1000 --lambda0 4550 --lambda1 5450 --step 2.6 --setup test1 --oprefix $PREF1
rvs_make_nd  --setup test1 --prefix $PREF1
rvs_make_ccf  --setup test1 --prefix $PREF1 --lambda0 4550 --lambda1 5450 --step 2  --every 2 --oprefix $PREF1

#!/bin/bash -e
DBFILE=test.db
PREF=tmp/
PREF1=templ_data_test/
python mktemps.py $PREF wave.fits
rm -f $DBFILE
rvs_read_grid  --prefix tmp --templdb $DBFILE
rvs_make_interpol --templdb $DBFILE --wavefile wave.fits --templprefix $PREF  --resol 1000 --lambda0 4550 --lambda1 5450 --step 1 --setup test --oprefix $PREF1
rvs_make_nd  --setup test --prefix $PREF1
rvs_make_ccf  --setup test --prefix $PREF1 --lambda0 4550 --lambda1 5450 --step 10  --every 2 --oprefix $PREF1

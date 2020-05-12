#!/bin/bash -e
DBFILE=test1.db
PREF=tmp1/
PREF1=templ_data_test/
mkdir -p $PREF/specs
mkdir -p $PREF1
rm -f $PREF/xx*fits
python mktemps_grid.py $PREF wave.fits
rm -f $DBFILE
rvs_read_grid  --prefix $PREF --templdb $DBFILE
rvs_make_interpol --templdb $DBFILE --wavefile $PREF/wave.fits --templprefix $PREF  --resol 1000 --lambda0 4550 --lambda1 5450 --step 1 --setup testgrid --oprefix $PREF1
rvs_make_nd  --regulargrid --setup testgrid --prefix $PREF1
rvs_make_ccf  --setup testgrid --prefix $PREF1 --lambda0 4550 --lambda1 5450 --step 10  --every 10 --oprefix $PREF1

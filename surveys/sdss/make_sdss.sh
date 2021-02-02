#!/bin/bash
set -e
NAME=sdss1
WAVE0=3500
WAVE1=9500
OUTPUT_PREFIX=templ_data
PHOENIXDIR=/home/skoposov/science/PHOENIX/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/
PHOENIXWAVE=/home/skoposov/science/PHOENIX/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits

# You may need this line if you haven't yet created the database of
# templates yet
# rvs_read_grid --prefix $PHOENIXDIR --templdb $OUTPUT_PREFIX/files.db

rvs_make_interpol --setup $NAME --lambda0 $WAVE0 --lambda1 $WAVE1 --resol 2000 --step 1 --templdb $OUTPUT_PREFIX/files.db --oprefix $OUTPUT_PREFIX --templprefix $PHOENIXDIR --wavefile $PHOENIXWAVE

rvs_make_nd --prefix $OUTPUT_PREFIX --setup $NAME

rvs_make_ccf --setup $NAME --lambda0 $WAVE0 --lambda1 $WAVE1 --every 30 --vsinis 0,300 --prefix $OUTPUT_PREFIX --oprefix $OUTPUT_PREFIX --step 1

#!/bin/sh
PREFIX=../../templ_data
TEMPLPREF=/home/skoposov/science/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/
WAVEFILE=/home/skoposov/science/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
RLAM0=5500
RLAM1=7900
BLAM0=3300
BLAM1=6100
ZLAM0=7300
ZLAM1=10000
VSINIS=0,300

python ../make_interpol.py --setup desi_r --lambda0 $RLAM0 --lambda1 $RLAM1 --resol 3700 --step 0.5 --templdb ${PREFIX}/files.db --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $WAVEFILE

python ../make_nd.py --prefix ${PREFIX}/ --setup desi_r

python ../make_ccf.py --setup desi_r --lambda0 $RLAM0 --lambda1 $RLAM1 --every 30 --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX} --step 0.5


python ../make_interpol.py --setup desi_b --lambda0 $BLAM0 --lambda1 $BLAM1 --resol 2800 --step 0.4 --templdb ${PREFIX}/files.db --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $WAVEFILE

python ../make_nd.py --prefix ${PREFIX}/ --setup desi_b

python ../make_ccf.py --setup desi_b --lambda0 $BLAM0 --lambda1 $BLAM1  --every 30 --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX} --step 0.4


python ../make_interpol.py --setup desi_z --lambda0 $ZLAM0 --lambda1 $ZLAM1 --resol 4200 --step 0.5 --templdb ${PREFIX}/files.db --oprefix ${PREFIX}/ --templprefix $TEMPLPREF --wavefile $WAVEFILE

python ../make_nd.py --prefix ${PREFIX}/ --setup desi_z

python ../make_ccf.py --setup desi_z --lambda0 $ZLAM0 --lambda1 $ZLAM1  --every 30 --vsinis $VSINIS --prefix ${PREFIX}/ --oprefix=${PREFIX}x --step 0.5

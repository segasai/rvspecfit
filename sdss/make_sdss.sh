NAME=sdss1
WAVE0=3500
WAVE1=9500

python make_interpol.py --setup $NAME --lambda0 $WAVE0 --lambda1 $WAVE1 --resol 2000 --step 1 --templdb ../templ_data/files.db --oprefix ../templ_data/ --templprefix /home/skoposov/science/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/ --wavefile /home/skoposov/science/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits

python make_nd.py --prefix ../templ_data/ --setup $NAME

python make_ccf.py --setup $NAME --lambda0 $WAVE0 --lambda1 $WAVE1 --every 30 --vsinis 0,300 --prefix ../templ_data/ --oprefix=../templ_data --step 1

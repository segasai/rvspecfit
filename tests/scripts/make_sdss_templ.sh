DIR=./templ_data_sdss
rvs_make_interpol --setup sdss1 --lambda0 3500 --lambda1 9500 --resol 2000 --step 1 --templdb ./templates_sdss.db --oprefix $DIR --templprefix /home/skoposov/science/PHOENIX/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/ --wavefile /home/skoposov/science/PHOENIX/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
rvs_make_nd --prefix $DIR --setup sdss1
rvs_make_ccf --setup sdss1 --lambda0 3500 --lambda1 9500 --every 30 --vsinis 0,300 --prefix $DIR --oprefix=$DIR --step 1

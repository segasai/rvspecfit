
rvs_make_interpol --setup desi_b --lambda0 3500 --lambda1 5900 --resol_func 'x/1.55' --step 0.4 --templdb ./files_sdss.db --oprefix ./xtmp_desi/ --templprefix /home/skoposov/science/PHOENIX/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/ --wavefile /home/skoposov/science/PHOENIX/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
rvs_make_nd --prefix ./xtmp_desi/ --setup desi_b
rvs_make_ccf --setup desi_b --lambda0 3500 --lambda1 5900 --every 30 --vsinis 0,300 --prefix ./xtmp_desi/ --oprefix=./xtmp_desi --step 0.4

rvs_make_interpol --setup desi_r --lambda0 5660 --lambda1 7720 --resol_func 'x/1.55' --step 0.4 --templdb ./files_sdss.db --oprefix ./xtmp_desi/ --templprefix /home/skoposov/science/PHOENIX/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/ --wavefile /home/skoposov/science/PHOENIX/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
rvs_make_nd --prefix ./xtmp_desi/ --setup desi_r
rvs_make_ccf --setup desi_r --lambda0 5660 --lambda1 7720 --every 30 --vsinis 0,300 --prefix ./xtmp_desi/ --oprefix=./xtmp_desi --step 0.4

rvs_make_interpol --setup desi_z --lambda0 7420 --lambda1 9924 --resol_func 'x/1.8' --step 0.4 --templdb ./files_sdss.db --oprefix ./xtmp_desi/ --templprefix /home/skoposov/science/PHOENIX/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/ --wavefile /home/skoposov/science/PHOENIX/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
rvs_make_nd --prefix ./xtmp_desi/ --setup desi_z
rvs_make_ccf --setup desi_z --lambda0 7420 --lambda1 9924 --every 30 --vsinis 0,300 --prefix ./xtmp_desi/ --oprefix=./xtmp_desi --step 0.4

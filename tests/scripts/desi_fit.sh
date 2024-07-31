#!/bin/bash -e

rvs_desi_fit  --output_dir ./tests_desi_output --no_subdirs  --minsn=2 --config tests/yamls/config_desi.yaml --objtypes='SCND_ANY,MWS_ANY,STD_*' --zbest_include --nthreads=2 tests/data/coadd-sv1-bright-10378.fits --process_status_file /tmp/tests_desi.status  --log_level=DEBUG

# 1 thread for coverage
rvs_desi_fit  --output_dir ./tests_desi_output --no_subdirs --minsn=2 --config tests/yamls/config_desi.yaml --objtypes='SCND_ANY,MWS_ANY,STD_*' --throw_exceptions --zbest_include --nthreads=1 tests/data/coadd-sv1-bright-10378.fits --process_status_file /tmp/tests_desi.status  --log_level=DEBUG


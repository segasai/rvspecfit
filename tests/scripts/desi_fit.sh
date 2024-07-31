#!/bin/bash -e

COV='echo'
RVS_FIT="$COV `command -v rvs_desi_fit`"

$RVS_FIT  --output_dir ./tests_desi_output --no_subdirs  --minsn=2 --config tests/yamls/config_desi.yaml --objtypes='SCND_ANY,MWS_ANY,STD_*' --zbest_include --nthreads=2 --process_status_file /tmp/tests_desi.status  --log_level=DEBUG tests/data/coadd-sv1-bright-10378.fits 

# 1 thread for coverage
$RVS_FIT  --output_dir ./tests_desi_output --no_subdirs --minsn=2 --config tests/yamls/config_desi.yaml --objtypes='SCND_ANY,MWS_ANY,STD_*' --throw_exceptions --zbest_include --zbest_select --nthreads=1 --process_status_file /tmp/tests_desi.status  --log_level=DEBUG tests/data/coadd-sv1-bright-10378.fits 

# specific targetids also plotting and zbest_select
$RVS_FIT  --output_dir ./tests_desi_output --no_subdirs --minsn=2 --config tests/yamls/config_desi.yaml --objtypes='SCND_ANY,MWS_ANY,STD_*' --throw_exceptions --zbest_include --nthreads=1 --process_status_file /tmp/tests_desi.status  --log_level=DEBUG --targetid=39628422527323686,39628422531515839 --doplot tests/data/coadd-sv1-bright-10378.fits


echo tests/data/coadd-sv1-bright-10378.fits > /tmp/queue.list
# queue
$RVS_FIT  --output_dir ./tests_desi_output --no_subdirs --minsn=2 --config tests/yamls/config_desi.yaml --objtypes='SCND_ANY,MWS_ANY,STD_*' --throw_exceptions --zbest_include --nthreads=1 --process_status_file /tmp/tests_desi.status  --log_level=DEBUG --queue_file=/tmp/queue.list

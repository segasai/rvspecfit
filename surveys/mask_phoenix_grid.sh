#!/bin/bash -e

# This script masks the known problematic spectra in the PHOENIX grid
# specifically the alpha=-0.4 cool stars and a couple of spectra 
# in the middle of the grid
# the arguments are input sql output sql file 


set -e -u 
if [ -f $1 ] ; then
    {
	cp $1 $2
	echo '
update files set bad=true where (alpha+0.4)<0.01 and teff<4500;
update files set bad=true where 
       abs(teff-3100)<1 and abs(logg-3)<0.01    and 
       abs(feh+.5)<0.01 and  abs(alpha-1.2)<0.01 ;
update files set bad=true where 
       abs(teff-3700)<1 and abs(logg-4)<0.01    and 
       abs(feh-.5)<0.01 and  abs(alpha-1.2)<0.01 ;

update files set bad=true where 
       abs(teff-2500)<1 and abs(logg-3)<0.01    and 
       abs(feh-1)<0.01 and  abs(alpha-1.2)<0.01 ;

update files set bad=true where 
       abs(teff-2900)<1 and abs(logg-1.5)<0.01    and 
       abs(feh+1)<0.01 and  abs(alpha-0.6)<0.01 ;

update files set bad=true where 
       abs(teff-3000)<1 and abs(logg-2)<0.01    and 
       abs(feh+.5)<0.01 and  abs(alpha-0.6)<0.01 ;

update files set bad=true where 
       abs(teff-3000)<1 and abs(logg-2.5)<0.01    and 
       abs(feh-0)<0.01 and  abs(alpha-0.6)<0.01 ;

 '| sqlite3 $2
    }  ; else {
    echo ' the first argument needs to be an sqlite db'
    } ; fi

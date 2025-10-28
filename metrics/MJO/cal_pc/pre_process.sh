#!/bin/bash


module load cdo

datadir='/glade/campaign/univ/uchi0014/benchmark-dev/predictions/'

for ncfile in ${datadir}pangu*2019*ens_1.nc; do
	filename="$(basename "$ncfile")"
	echo $filename
    cdo -selvar,mean_top_net_long_wave_radiation_flux,u_component_of_wind $ncfile $filename.tmp	
	cdo -remapcon,144x72.txt $filename.tmp ./$filename.mjo_144x72
    rm -f $filename.tmp
done

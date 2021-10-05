#!/bin/bash

set -u
set -e

NPARALLEL=10

#rm features/*.csv

COUNTER=0
for f in data/qps_200000/*dmesg*
do
    echo $f
    python featurizer.py $f ./features &
    #sleep 10 &
    if [[ $(($COUNTER % $NPARALLEL)) -eq 0 ]]
    then
	wait
    fi

    COUNTER=$(($COUNTER+1))
done

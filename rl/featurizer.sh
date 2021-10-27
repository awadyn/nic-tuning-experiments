#!/bin/bash

set -u
set -e

NPARALLEL=10

#rm features/*.csv

SRC_DIR='data/qps_200000/'
TARGET_DIR='./features'

COUNTER=0
for f in $SRC_DIR/*dmesg*
do
    echo $f
    python featurizer.py $f $TARGET_DIR &
    #sleep 10 &
    if [[ $(($COUNTER % $NPARALLEL)) -eq 0 ]]
    then
	   wait
    fi

    COUNTER=$(($COUNTER+1))
done

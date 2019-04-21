#!/usr/bin/env bash

SNAPSHOTS_ROOT=../../workdir/snapshots
DATA=../../Data/pre_processed/

for SNAPSHOT in $(ls ${SNAPSHOTS_ROOT} -t | grep "2019_");
do
    echo "===> Working on the snapshot ${SNAPSHOT}"
    python evaluate_metrics.py --snapshots_root ${SNAPSHOTS_ROOT} --snapshot ${SNAPSHOT} --dataset_dir ${DATA}
done


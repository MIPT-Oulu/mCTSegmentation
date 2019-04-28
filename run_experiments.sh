#!/usr/bin/env bash

sh download_data.sh

cd scripts/

# Experiment 1
python train.py --loss bce

# Experiment 2
python train.py --loss jaccard

sh ../evaluate_snapshots.sh
#!/usr/bin/env bash

# Getting the data
#sh download_data.sh

# Running the experiments

## Experiment 1
#python scripts/train.py --loss bce

## Experiment 2
#python scripts/train.py --loss jaccard

## Experiment 3
python scripts/train.py --loss combined


sh evaluate_snapshots.sh

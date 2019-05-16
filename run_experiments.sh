#!/usr/bin/env bash

# Getting the data
sh download_data.sh

# Running the experiments

## Experiment 1
python code/train.py --loss bce

## Experiment 2
#python code/train.py --loss jaccard

## Experiment 3
python code/train.py --loss focal

## Experiment 4
#python code/train.py --loss jaccard --log_jaccard True

## Experiment 5
#python code/train.py --loss combined

## Experiment 6
python code/train.py --loss combined --log_jaccard True

sh evaluate_snapshots.sh

python code/generate_pictures_and_tables.py
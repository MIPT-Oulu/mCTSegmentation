#!/usr/bin/env bash

# Getting the data
sh download_data.sh

# Running the experiments

## Experiment 1
python scripts/train.py --loss bce

## Experiment 2
python scripts/train.py --loss jaccard

## Experiment 3
python scripts/train.py --loss focal

## Experiment 4
python scripts/train.py --loss jaccard --log_jaccard True

## Experiment 5
python scripts/train.py --loss combined

## Experiment 6
python scripts/train.py --loss combined --log_jaccard True

sh evaluate_snapshots.sh

python scripts/generate_pictures_and_tables.py
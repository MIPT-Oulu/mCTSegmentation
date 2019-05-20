#!/usr/bin/env bash

# Getting the data
sh download_data.sh

# Running the experiments

## Experiment 1
python code/train.py --loss bce
## Experiment 2
python code/train.py --loss focal
## Experiment 3
python code/train.py --loss combined --log_jaccard True

sh evaluate_snapshots.sh

python code/generate_pictures_and_tables.py --extension png
python code/generate_pictures_and_tables.py --extension pdf

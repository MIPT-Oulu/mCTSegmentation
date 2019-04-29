#!/usr/bin/env bash

CUR_DIR=$(pwd)
DATA_DIR_LOC=../

mkdir -p $DATA_DIR_LOC
cd $DATA_DIR_LOC
wget http://mipt-ml.oulu.fi/data/mCTSegmentation/PTA_dataset.tar.gz
tar -xvf PTA_dataset.tar.gz


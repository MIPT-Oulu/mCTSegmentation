#!/usr/bin/env bash

DATA_DIR_LOC=../

CUR_DIR=$(pwd)

mkdir -p $DATA_DIR_LOC
cd $DATA_DIR_LOC
wget http://mipt-ml.oulu.fi/data/mCTSegmentation/PTA_dataset.tar.gz
tar -xvf PTA_data.tar.gz
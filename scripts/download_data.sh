#!/usr/bin/env bash

DATA_DIR=../../Data/

CUR_DIR=$(pwd)

cd $DATA_DIR
wget http://mipt-ml.oulu.fi/data/mCTSegmentation/PTA_data.tar.gz
tar -xvf PTA_data.tar.gz
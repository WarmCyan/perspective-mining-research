#!/bin/bash

pushd ../src

preprocess_kaggle.py -i ../data/raw/kaggle1 -o ../data/cache/kaggle1_preprocessed.dat

popd

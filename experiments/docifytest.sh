#!/bin/bash

pushd ../perspective

docify.py -i ../data/raw/kaggle1 -o ../data/cache/kaggle1_docs.dat

popd

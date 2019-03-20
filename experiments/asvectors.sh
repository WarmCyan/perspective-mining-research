#!/bin/bash

pushd ../perspective

docify.py -i ../data/raw/kaggle1 -o ../data/cache/kaggle1_docs.json


pushd aspect_detection/bootstrap

detection.py -i ../../../data/cache/kaggle1_docs.json -o ../../../data/cache/kaggle_aspects -c 500 --overwrite

popd

pushd sentiment_analysis
as_vec.py -i ../../data/cache/kaggle_aspects -o ../../data/cache/as_vectors --overwrite
popd

popd

#!/bin/bash

pushd ../perspective

docify.py -i ../data/raw/kaggle1 -o ../data/cache/kaggle1_docs.json


pushd aspect_detection/bootstrap

detection.py -i ../../../data/cache/kaggle1_docs.json -o ../../../data/cache/kaggle_aspects -c 100 

popd

pushd sentiment_analysis
as_vec.py -i ../../data/cache/kaggle_aspects -o ../../data/cache/as_vectors --overwrite
popd

popd

#!/bin/bash

pushd ../perspective

docify.py -i ../data/raw/kaggle1 -o ../data/cache/kaggle1_docs.json


pushd aspect_detection/bootstrap

detection.py -i ../../../data/cache/kaggle1_docs.json -o ../../../data/cache/kaggle_aspects.json -c 100 --overwrite

popd

popd

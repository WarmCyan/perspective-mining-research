#!/bin/bash

pushd ../src

docify.py -i ../data/raw/kaggle1 -o ../data/cache/kaggle1_docs.json


pushd aspect_detection/bootstrap

detection.py -i ../../../data/cache/kaggle1_docs.json -c 10

popd

popd

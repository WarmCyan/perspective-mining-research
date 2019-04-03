#!/bin/bash

pushd ../perspective

prune_articles.py -i ../data/raw/webhose_political/news.csv -o ../data/cache/webhose/webhose.csv
docify.py -i ../data/cache/webhose -o ../data/cache/webhose.json --col text

pushd aspect_detection/bootstrap
detection.py -i ../../../data/cache/webhose.json -o ../../../data/cache/webhose_aspects -c 100 --overwrite
popd

pushd sentiment_analysis
as_vec.py -i ../../data/cache/webhose_aspects -o ../../data/cache/webhose_as_vectors --overwrite
popd

popd

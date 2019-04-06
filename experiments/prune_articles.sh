#!/bin/bash

pushd ../perspective

prune_articles.py -i ../data/raw/webhose_political/news.csv -o ../data/cache/webhose.csv

popd


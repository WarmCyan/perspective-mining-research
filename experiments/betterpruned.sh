
#!/bin/bash

pushd ../perspective

prune_articles.py -i ../data/raw/webhose_political/news.csv -o ../data/cache/webhose_pruned/webhose_pruned.csv --thresh_source 200 --thresh_words 200 --overwrite
docify.py -i ../data/cache/webhose_pruned -o ../data/cache/webhose_pruned.json --col text --overwrite

pushd aspect_detection/bootstrap
detection.py -i ../../../data/cache/webhose_pruned.json -o ../../../data/cache/webhose_pruned_aspects --support 0.005 -w 4 --overwrite
popd

pushd sentiment_analysis
as_vec.py -i ../../data/cache/webhose_pruned_aspects -o ../../data/cache/webhose_pruned_as_vectors --overwrite
popd

popd

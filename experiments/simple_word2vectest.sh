#!/bin/bash

pushd ../perspective

sentencify.py -i ../data/raw/kaggle1 -o ../data/cache/kaggle1_sentences.dat
word2vec.py -i ../data/cache/kaggle1_sentences.dat -o ../data/cache/kaggle1_w2v_model

popd

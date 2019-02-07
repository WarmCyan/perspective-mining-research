#!/bin/bash

pushd ../perspective

# 100,000,000
#sentencify.py -i ../data/raw/kaggle1 -o ../data/cache/kaggle1_sentences.dat -c 10000000000 --overwrite
sentencify.py -i ../data/raw/kaggle1 -o ../data/cache/kaggle1_sentences.dat
word2vec.py -i ../data/cache/kaggle1_sentences.dat -o ../data/cache/kaggle1_w2v_model

preprocess_kaggle.py -i ../data/raw/kaggle1 -o ../data/cache/kaggle1_preprocessed.dat -c 100 --overwrite

pushd aspect_detection/neural_attn_model
bash -c ". bin/activate && cd code && python train.py -o ../../../../data/cache/test -i ../../../../data/cache/kaggle1_preprocessed.dat --emb ../../../../data/cache/kaggle1_w2v_model -as 100"
popd

popd

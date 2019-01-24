#!/bin/bash

pushd ../src

sentencify.py -i ../data/raw/kaggle1 -o ../data/cache/kaggle1_sentences.dat -c 1000 --overwrite
word2vec.py -i ../data/cache/kaggle1_sentences.dat -o ../data/cache/kaggle1_w2v_model --overwrite

preprocess_kaggle.py -i ../data/raw/kaggle1 -o ../data/cache/kaggle1_preprocessed.dat -c 100 --overwrite

pushd aspect_detection/neural_attn_model
bash -c ". bin/activate && cd code && python train.py -o ../../../../data/cache/test -i ../../../../data/cache/kaggle1_preprocessed.dat --emb ../../../../data/cache/kaggle1_w2v_model"
popd

popd

#!/bin/bash

./simple_word2vectest.sh

pushd ../src

preprocess_kaggle.py -i ../data/raw/kaggle1 -o ../data/cache/kaggle1_preprocessed.dat

pushd aspect_detection/neural_attn_model
bash -c ". bin/activate && cd code && python train.py -o ../../../../data/cache/test -i ../../../../data/cache/kaggle1_preprocessed.dat --emb ../../../../data/cache/kaggle1_w2v_model"
popd

popd

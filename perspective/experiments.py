#!/bin/python3

import pandas as pd
import argparse
import logging
import os
import json

import hashlib

from tqdm import tqdm
import utility

import docify

def run(experiment_path, raw_path, cache_path, overwite=False):

    experiment_list = []

    preprocess_hashes = {}
    vectorize_hashes = {}
    predict_hashes = {}

    index = 0
    for experiment in experiment_list:
        index += 1
        logging.info("Starting experiment %i - %s", index, json.dumps(experiment))

        # preprocessing
        preprocess_params = experiment[0]
        preprocess_hash = hash_params(preprocess_params)
        logging.info("Running preprocessing for experiment %i - %s - %s", index, preprocess_hash, json.dumps(preprocess_params))
                
        if preprocess_hash in preprocess_params.keys():
            preprocess_hashes[preprocess_hash] = preprocess_params
        preprocess(**preprocess_params)

        # vectorizing
        vectorize_params = experiment[1]
        vectorize_hash = hash_params(vectorize_params)
        logging.info("Running vectorization for experiment %i - %s - %s", index, vectorize_hash, json.dumps(vectorize_params))
        
        if vectorize_hash in vectorize_params.keys():
            vectorize_hashes[vectorize_hash] = vectorize_params
        vectorize(**vectorize_params)

        predict_params = experiment[1]
        predict_hash = hash_params(predict_params)
        
        if predict_hash in predict_params.keys():
            predict_hashes[predict_hash] = predict_params
        logging.info("Running prediction for experiment %i - %s - %s", index, vectorize_hash, json.dumps(predict_params))
        predict(**predict_params)

def hash_params(params):
    if "name" in params.keys():
        return params["name"]
    return hashlib.md5(json.dumps(params, sort_keys=True))
    


def preprocess(preprocess_hash, experiment_path, data_folder, count=-1, keywords=[], ignore_source=[], content_column="content", source_content="publication", **kwargs):
    docify.docify(data_folder, experiment_path + preprocess_hash + "_documents.json")


# methods: tfidf, as_vec
def vectorize(method="tfidf", **kwargs):
    pass


def predict():
    pass


def parse():
    """Handle all command line argument parsing.

    Returns the parsed args object from the parser
    """
    parser = argparse.ArgumentParser()
    parser = utility.add_common_parsing(parser)

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    ARGS = parse()
    utility.init_logging(ARGS.log_path)
    raw_path, cache_path = utility.fix_paths(ARGS.experiment_path, ARGS.input_path, ARGS.output_path)

    run(ARGS.experiment_path, raw_path, cache_path, ARGS.overwrite)

#!/bin/python3

import pandas as pd
import argparse
import logging
import json
import shutil
import os
import datetime

import hashlib

from tqdm import tqdm
import utility

import docify
import tokenization
import aspect_detection
import sentiment_analysis
import tfidfify
import combine


THREAD_COUNT = 4


# in this context overwrite means re-run existing experiments
def run(experiment_path, raw_path, cache_path, overwrite=False):

    experiment_list = [] # populate this from some other folder

    preprocess_hashes = {}
    vectorize_hashes = {}
    predict_hashes = {}

    results_list = []

    results_path = experiment_path + "/results.csv"

    # search for previous results file, and load them in
    if os.path.exists(results_path):
        old_results_df = pd.read_csv(results_path)
        results_list = old_results_df.to_dict(orient='records')

    # iterate through each experiment
    index = 0
    for experiment in experiment_list:
        index += 1
        logging.info("################################################")
        logging.info("Starting experiment %i - %s", index, json.dumps(experiment))
        logging.info("################################################")

        # get experiment parameter hashes
        preprocess_params = experiment["preprocess"]
        preprocess_hash = hash_params(preprocess_params)
        vectorize_params = experiment["vectorize"]
        vectorize_hash = hash_params(vectorize_params)
        predict_params = experiment["predict"]
        predict_hash = hash_params(predict_params)
        
        # determine output folders for each stage
        preprocess_folder = experiment_path + "/preproc_" + preprocess_hash
        vectorize_folder = experiment_path + "/vec_" + vectorize_hash
        predict_folder = experiment_path + "/pred_" + predict_hash

        # check if experiment already run
        exists = False
        if old_results_df[(old_results_df.preprocess == preprocess_hash) & (old_results_df.vectorize == vectorize_hash) & (old_results_df.predict == predict_hash)].shape[0] > 0:
            exists = True
        
        if exists:
            logging.info("Previous result from experiment found...")
            if overwrite or ("rerun" in experiment.keys() and experiment["rerun"]):
                logging.info("Skipping")
                continue
            else:
                logging.info("Deleting previous experiment components...")
                shutil.rmtree(preprocess_folder, ignore_errors=True)
                shutil.rmtree(vectorize_folder, ignore_errors=True)
                shutil.rmtree(predict_folder, ignore_errors=True)

        # ---------------------
        # PREPROCESSING
        # ---------------------
        logging.info("Running preprocessing for experiment %i - %s - %s", index, preprocess_hash, json.dumps(preprocess_params))

        # make the output path
        if not os.path.isdir(preprocess_folder):
            os.makedirs(preprocess_folder)

        # add to the list of hashes
        if preprocess_hash in preprocess_params.keys():
            preprocess_hashes[preprocess_hash] = preprocess_params

        # run the preprocessing!
        preprocess(preprocess_folder, **preprocess_params)

        # ---------------------
        # VECTORIZING
        # ---------------------
        logging.info("Running vectorization for experiment %i - %s - %s", index, vectorize_hash, json.dumps(vectorize_params))
        
        # make the output path
        if not os.path.isdir(vectorize_folder):
            os.makedirs(vectorize_folder)
            
        # add to the list of hashes
        if vectorize_hash in vectorize_params.keys():
            vectorize_hashes[vectorize_hash] = vectorize_params

        # run the vectorizing!
        vectorize(preprocess_folder, vectorize_folder, **vectorize_params)

        # ---------------------
        # PREDICTING
        # ---------------------
        logging.info("Running prediction for experiment %i - %s - %s", index, vectorize_hash, json.dumps(predict_params))
        
        # make the output path
        if not os.path.isdir(predict_folder):
            os.makedirs(predict_folder)
            
        # add to the list of hashes
        if predict_hash in predict_params.keys():
            predict_hashes[predict_hash] = predict_params

        # run the prediction!
        result = predict(**predict_params)

        # gather result information
        result.update(preprocess_params)
        result.update(vectorize_params)
        result.update(predict_params)
        result.update({"preprocess":preprocess_hash, "vectorize":vectorize_hash, "predict":predict_hash})
        result.update({"index":index})

        results_list.append(result)

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(results_path)
        results_df.to_csv(experiment_path + "/results_" + datetime.datetime.now().strftime("%Y-%m-%d"))


def hash_params(params):
    if "name" in params.keys():
        return params["name"]
    return hashlib.md5(json.dumps(params, sort_keys=True))
    


# data_folder
# document_count (-1)
# content_column ("content")
# source_column ("publication")
# keywords ([])
# ignore_sources ([])
def preprocess(preprocess_folder, **kwargs):
    documents_file = preprocess_folder + "/documents.json"
    
    docify.docify(input_folder=kwargs.get("data_folder"), output_path=documents_file, count=kwargs.get("document_count", -1), content_column=kwargs.get("content_column", "content"), source_column=kwargs.get("source_column", "publication"), keywords=kwargs.get("keywords", []), ignore_source=kwargs.get("ignore_sources", []))
    
    tokenization.tokenize(input_file=documents_file, output_path=(preprocess_folder + "/tokens"))


# support
# target_aspect_count (-1)
# ner (False)
# minimum_flr (10.0)
# sentiment_distance_dist_sd (1)
def vectorize(preprocess_folder, vectorize_folder, **kwargs):
    documents_file = preprocess_folder + "/documents.json"
    tokens_folder = preprocess_folder + "/tokens"

    tfidf_path = vectorize_folder + "/tfidf.json"

    aspect_data_path = vectorize_folder + "/aspects"

    as_vec_path = vectorize_folder + "/as_vec"
    
    aspect_detection.bootstrap.detection.detect(input_path=tokens_folder, output_path=aspect_data_path, support=kwargs.get("support"), target_count=kwargs.get("target_aspect_count", -1), thread_count=THREAD_COUNT, named_entity_recog=kwargs.get("ner", False))

    tfidfify.tfidf(input_path=documents_file, output_path=tfidf_path)

    sentiment_analysis.as_vec.create_as_vectors(input_path=aspect_data_path, tokens_path=tokens_folder, output_path=as_vec_path, minimum_flr=kwargs.get("minimum_flr", 10.0), sd=kwargs.get("sentiment_distance_dist_sd", 1))

    combine.combine(input_file_1=tfidf_path, input_file_2=(as_vec_path + "/doc_as_vectors.json"), output_file=(vectorize_folder + "/tfidf_as_vec_combined.json"))
    

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

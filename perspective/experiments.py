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
import aspect_detection.bootstrap
import aspect_detection.bootstrap.detection
import sentiment_analysis
import sentiment_analysis.as_vec
import tfidfify
import combine
import prediction_model


THREAD_COUNT = 2

# in this context overwrite means re-run existing experiments
def run(experiment_path, raw_path, cache_path, overwrite=False):


    preprocess_climate = dict(data_folder=(raw_path+"/kaggle1"), document_count=5000, keywords=["climate change","global warming","climate"], ignore_sources=["CNN","Buzzfeed News"])

    
    experiment_list = [
        # non-named entity recognition
        {
            "preprocess":preprocess_climate,
            "vectorize":dict(support=0.01, ner=False, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False),
            "predict":dict(source="as_vec", undersample=False, oversample=False, model_type="lr", class_balance=False)
        },
        {
            "preprocess":preprocess_climate,
            "vectorize":dict(support=0.01, ner=False, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False),
            "predict":dict(source="tfidf", undersample=False, oversample=False, model_type="lr", class_balance=False)
        },
        {
            "preprocess":preprocess_climate,
            "vectorize":dict(support=0.01, ner=False, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False),
            "predict":dict(source="combined", undersample=False, oversample=False, model_type="lr", class_balance=False)
        },

        # named entity recognition
        {
            "preprocess":preprocess_climate,
            "vectorize":dict(support=0.01, ner=True, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False),
            "predict":dict(source="as_vec", undersample=False, oversample=False, model_type="lr", class_balance=False)
        },
        {
            "preprocess":preprocess_climate,
            "vectorize":dict(support=0.01, ner=True, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False),
            "predict":dict(source="tfidf", undersample=False, oversample=False, model_type="lr", class_balance=False)
        },
        {
            "preprocess":preprocess_climate,
            "vectorize":dict(support=0.01, ner=True, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False),
            "predict":dict(source="combined", undersample=False, oversample=False, model_type="lr", class_balance=False)
        },
        
        # lower support
        {
            "preprocess":preprocess_climate,
            "vectorize":dict(support=0.0001, ner=True, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False),
            "predict":dict(source="as_vec", undersample=False, oversample=False, model_type="lr", class_balance=False)
        },
        {
            "preprocess":preprocess_climate,
            "vectorize":dict(support=0.0001, ner=True, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False),
            "predict":dict(source="combined", undersample=False, oversample=False, model_type="lr", class_balance=False)
        },
        
        # matched tfidf
        {
            "preprocess":preprocess_climate,
            "vectorize":dict(support=0.01, ner=True, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=True),
            "predict":dict(source="tfidf", undersample=False, oversample=False, model_type="lr", class_balance=False)
        },
        {
            "preprocess":preprocess_climate,
            "vectorize":dict(support=0.01, ner=True, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False, tfidf_feature_count=422),
            "predict":dict(source="tfidf", undersample=False, oversample=False, model_type="lr", class_balance=False)
        },
    ]

    preprocess_hashes = {}
    vectorize_hashes = {}
    predict_hashes = {}

    results_list = []

    results_path = experiment_path + "/results.csv"

    # search for previous results file, and load them in
    old_results_df = None
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
        if old_results_df is not None and old_results_df[(old_results_df.preprocess == preprocess_hash) & (old_results_df.vectorize == vectorize_hash) & (old_results_df.predict == predict_hash)].shape[0] > 0:
            exists = True
        
        if exists:
            logging.info("Previous result from experiment found...")
            if overwrite or ("rerun" in experiment.keys() and experiment["rerun"]):
                logging.info("Deleting previous experiment components...")
                shutil.rmtree(preprocess_folder, ignore_errors=True)
                shutil.rmtree(vectorize_folder, ignore_errors=True)
                shutil.rmtree(predict_folder, ignore_errors=True)
            else:
                logging.info("Skipping")
                continue
        else:
            if overwrite:
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
        result = {"score": predict(preprocess_folder, vectorize_folder, predict_folder, **predict_params)}

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
    return str(hashlib.md5(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest())
    


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
# tfidf_feature_count (5000)
# tfidf_match_vocab (False)
def vectorize(preprocess_folder, vectorize_folder, **kwargs):
    documents_file = preprocess_folder + "/documents.json"
    tokens_folder = preprocess_folder + "/tokens"

    tfidf_path = vectorize_folder + "/tfidf.json"

    aspect_data_path = vectorize_folder + "/aspects"

    match_vocab = None
    match_vocab_arg = kwargs.get("tfidf_match_vocab", False)
    if match_vocab_arg:
        match_vocab = aspect_data_path + "/aspects.json"
    
    if not os.path.isdir(aspect_data_path):
        os.makedirs(aspect_data_path)

    as_vec_path = vectorize_folder + "/as_vec"
    
    aspect_detection.bootstrap.detection.detect(input_path=tokens_folder, output_path=aspect_data_path, support=kwargs.get("support"), target_count=kwargs.get("target_aspect_count", -1), thread_count=THREAD_COUNT, named_entity_recog=kwargs.get("ner", False))

    tfidfify.tfidf(input_path=documents_file, output_path=tfidf_path, feature_count=kwargs.get("tfidf_feature_count", 5000), match_vocab=match_vocab)

    sentiment_analysis.as_vec.create_as_vectors(input_path=aspect_data_path, tokens_path=tokens_folder, output_path=as_vec_path, minimum_flr=kwargs.get("minimum_flr", 10.0), sd=kwargs.get("sentiment_distance_dist_sd", 1))

    combine.combine(input_file_1=tfidf_path, input_file_2=(as_vec_path + "/doc_as_vectors.json"), output_file=(vectorize_folder + "/tfidf_as_vec_combined.json"))
    

# source
# undersample (False)
# oversample (False)
# model_type ("lr")
# class_balance (False)
def predict(preprocess_folder, vectorize_folder, predict_folder, **kwargs):
    documents_file = preprocess_folder + "/documents.json"

    source = kwargs.get("source")
    input_file = ""
    if source == "tfidf":
        input_file = vectorize_folder + "/tfidf.json"
    elif source == "as_vec":
        input_file = vectorize_folder + "/as_vec/doc_as_vectors.json"
    elif source == "combined":
        input_file = vectorize_folder + "/tfidf_as_vec_combined.json"
    
    score = prediction_model.predict(input_file=input_file, output_path=predict_folder, document_set=documents_file, model_type=kwargs.get("model_type", "lr"), undersample=kwargs.get("undersample", False), oversample=kwargs.get("oversample", False), class_balance=kwargs.get("class_balance", True))


    return score
    


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

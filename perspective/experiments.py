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
import add_lean


THREAD_COUNT = 2

# in this context overwrite means re-run existing experiments
def run(experiment_path, raw_path, cache_path, overwrite=False):

    preprocess_climate = dict(data_folder=(raw_path+"/kaggle1"), document_count=5000, keywords=["climate change","global warming","climate"], ignore_sources=["CNN","Buzzfeed News"])
    preprocess_all = dict(data_folder=(raw_path+"/kaggle1"), document_count=5000, keywords=[], ignore_sources=["CNN","Buzzfeed News"])
    vectorize_normal = dict(support=0.01, ner=False, noun_phrases=True, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False)
    vectorize_normal_ner = dict(support=0.01, ner=True, noun_phrases=False, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False)
    vectorize_normal_ner_and = dict(support=0.01, ner=True, noun_phrases=True, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False)
    vectorize_normal_lean = dict(support=0.01, ner=False, noun_phrases=True, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False, lean=True)
    vectorize_normal_ner_lean = dict(support=0.01, ner=True, noun_phrases=False, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False, lean=True)
    vectorize_normal_ner_and_lean = dict(support=0.01, ner=True, noun_phrases=True, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False, lean=True)


    def generate_model_type_experiments(experiment_list):
        new_experiments = {}
        for experiment in experiment_list:
            if "model_type" not in experiment["predict"].keys():
                new_experiment = dict(experiment)
                experiment["predict"]["model_type"] = "lr"
                new_experiment["predict"]["model_type"] = "nb"
                experiment_list.append(new_experiment)
        return experiment_list

    def generate_source_experiments(experiment_list):
        new_experiments = {}
        for experiment in experiment_list:
            if "source" not in experiment["predict"].keys():
                new_experiment1 = dict(experiment)
                new_experiment2 = dict(experiment)
                experiment["predict"]["source"] = "as_vec"
                new_experiment1["predict"]["source"] = "tfidf"
                new_experiment2["predict"]["source"] = "combined"
                experiment_list.append(new_experiment1)
                experiment_list.append(new_experiment2)
        return experiment_list

    def generate_flr_experiments(experiment_list):
        new_experiments = {}
        for experiment in experiment_list:
            if "minimum_flr" not in experiment["vectorize"].keys():
                new_experiment1 = dict(experiment)
                experiment["vectorize"]["minimum_flr"] = 10.0
                new_experiment1["vectorize"]["minimum_flr"] = 100.0
                experiment_list.append(new_experiment1)
        return experiment_list

    def generate_sentiment_top_only_experiments(experiment_list):
        new_experiments = {}
        for experiment in experiment_list:
            if "top_only" not in experiment["vectorize"].keys():
                new_experiment1 = dict(experiment)
                experiment["vectorize"]["top_only"] = False
                new_experiment1["vectorize"]["top_only"] = True
                experiment_list.append(new_experiment1)
        return experiment_list

    def generate_preprocess(experiment_list):
        new_experiments = {}
        for experiment in experiment_list:
            if "preprocess" not in experiment.keys():
                new_experiment1 = dict(experiment)
                experiment["preprocess"] = preprocess_climate
                new_experiment1["preprocess"] = preprocess_all
                experiment_list.append(new_experiment1)
        return experiment_list
        

    experiment_list = [
        {
            "vectorize":dict(support=0.05, ner=False, noun_phrases=True, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False),
            "predict":dict(undersample=False, oversample=False, class_balance=False)
        },
        
        # non-named entity recognition
        {
            "vectorize":vectorize_normal,
            "predict":dict(undersample=False, oversample=False, class_balance=False)
        },

        # named entity recognition
        {
            "vectorize":vectorize_normal_ner,
            "predict":dict(undersample=False, oversample=False, class_balance=False)
        },
        
        # lower support
        {
            "vectorize":dict(support=0.005, ner=True, noun_phrases=False,  minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False),
            "predict":dict(undersample=False, oversample=False, class_balance=False)
        },
        
        # matched tfidf
        {
            "vectorize":dict(support=0.01, ner=True, noun_phrases=False, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=False, tfidf_feature_count=422),
            "predict":dict(source="tfidf", undersample=False, oversample=False, class_balance=False)
        },
        {
            "vectorize":dict(support=0.01, ner=True, noun_phrases=False, minimum_flr=10.0, sentiment_distance_dist_sd=1, tfidf_match_vocab=True),
            "predict":dict(source="tfidf", undersample=False, oversample=False, class_balance=False)
        },

        # ner and regular
        {
            "vectorize":vectorize_normal_ner_and,
            "predict":dict(undersample=False, oversample=False, class_balance=False)
        },

        # lean
        {
            "vectorize":vectorize_normal_lean,
            "predict":dict(undersample=False, oversample=False, class_balance=False, lean=True)
        },
        
        # ner and lean
        {
            "vectorize":vectorize_normal_ner_lean,
            "predict":dict(undersample=False, oversample=False, class_balance=False, lean=True)
        },
    ]

    experiment_list = generate_source_experiments(experiment_list)
    experiment_list = generate_model_type_experiments(experiment_list)
    experiment_list = generate_flr_experiments(experiment_list)
    experiment_list = generate_sentiment_top_only_experiments(experiment_list)
    experiment_list = generate_preprocess(experiment_list)

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
        logging.info("Starting experiment %i/%i - %s", index, len(experiment_list), json.dumps(experiment))
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
        vectorize(raw_path, preprocess_folder, vectorize_folder, **vectorize_params)

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
# noun_phrases (True)
# minimum_flr (10.0)
# sentiment_distance_dist_sd (1)
# top_only (False)
# tfidf_feature_count (5000)
# tfidf_match_vocab (False)
# lean (False)
def vectorize(raw_path, preprocess_folder, vectorize_folder, **kwargs):
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

    lean = kwargs.get("lean", False)

    as_vec_path = vectorize_folder + "/as_vec"

    if add_lean:
        add_lean.add_lean(input_file=documents_file, output_file=documents_file, bias_file=(raw_path + "/bias_data.json"))

    
    aspect_detection.bootstrap.detection.detect(input_path=tokens_folder, output_path=aspect_data_path, support=kwargs.get("support"), target_count=kwargs.get("target_aspect_count", -1), thread_count=THREAD_COUNT, named_entity_recog=kwargs.get("ner", False), noun_phrases=kwargs.get("noun_phrases", True))

    tfidfify.tfidf(input_path=documents_file, output_path=tfidf_path, feature_count=kwargs.get("tfidf_feature_count", 5000), match_vocab=match_vocab)

    sentiment_analysis.as_vec.create_as_vectors(input_path=aspect_data_path, tokens_path=tokens_folder, output_path=as_vec_path, minimum_flr=kwargs.get("minimum_flr", 10.0), sd=kwargs.get("sentiment_distance_dist_sd", 1), top_only=kwargs.get("top_only", False))

    combine.combine(input_file_1=tfidf_path, input_file_2=(as_vec_path + "/doc_as_vectors.json"), output_file=(vectorize_folder + "/tfidf_as_vec_combined.json"))
    

# source
# undersample (False)
# oversample (False)
# model_type ("lr")
# class_balance (False)
# lean (False))
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
    
    score = prediction_model.predict(input_file=input_file, output_path=predict_folder, document_set=documents_file, model_type=kwargs.get("model_type", "lr"), undersample=kwargs.get("undersample", False), oversample=kwargs.get("oversample", False), class_balance=kwargs.get("class_balance", True), predict_lean=kwargs.get("lean", False))


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

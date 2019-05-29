#!/usr/bin/env python

"""Attaches the predictions of the models and the articles table."""

import argparse
import datetime
import logging
import pandas as pd
import json
import pickle

from collections import OrderedDict

from sklearn.model_selection import train_test_split
import utility

def examine(input_file, output_path, document_set, model_path):
    logging.info("Creating examination of output probabilities")

    logging.info("Loading document set...")
    docs = pd.read_json(document_set)
    
    logging.info("Loading features...")
    features = pd.read_json(input_file)
    #with open(input_file, 'r') as infile:
        #features = json.load(infile)

    X = features
    y = docs
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    
    logging.info("Loading model...")
    clf = pickle.load(open(model_path), 'rb')

    logging.info("Running predictions...")
    predictions = clf.predict_proba(X_test)
    classes = clf.classes_

    predictions_df = pd.DataFrame(predictions, columns=classes)
    
    examine_df = pd.concat([y_test, predictions_df])

    logging.info("Storing examination table at %s", output_path)
    examine_df.to_csv(output_path)
    
def parse():
    """Handle all command line argument parsing.

    Returns the parsed args object from the parser
    """
    parser = argparse.ArgumentParser()
    parser = utility.add_common_parsing(parser)

    parser.add_argument(
        "--documents",
        dest="documents",
        type=str,
        required=False,
        default="",
        metavar="<str>",
        help="the document set",
    )
    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        required=False,
        default="",
        metavar="<str>",
        help="the model",
    )

    
    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    ARGS = parse()
    utility.init_logging(ARGS.log_path)
    input_path, output_path = utility.fix_paths(ARGS.experiment_path, ARGS.input_path, ARGS.output_path)
    documents_path, output_path = utility.fix_paths(ARGS.experiment_path, ARGS.documents, ARGS.output_path)
    model_path, output_path = utility.fix_paths(ARGS.experiment_path, ARGS.model, ARGS.output_path)

    examine(input_path, output_path, documents_path, model_path)

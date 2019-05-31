#!/usr/bin/env python

"""Attaches the predictions of the models and the articles table."""

import argparse
import datetime
import logging
import math
import pandas as pd
import json
import pickle

from collections import OrderedDict

from sklearn.model_selection import train_test_split
import utility

# expects a row (array) of probabilities
def prediction_entropy(row):
    entropy = 0
    for col in row:
        entropy += col * math.log10(col)

    entropy *= -1
    return entropy

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
    clf = pickle.load(open(model_path, 'rb'))

    logging.info("Running predictions...")
    predictions = clf.predict_proba(X_test)
    static_predictions = clf.predict(X_test)
    classes = clf.classes_

    predictions_df = pd.DataFrame(predictions, index=X_test.index, columns=classes)

    entropy_col_df = pd.DataFrame(predictions_df.apply(lambda row: prediction_entropy(row), axis=1), index=X_test.index, columns=["entropy"])
    predictions_df = pd.concat([predictions_df, entropy_col_df], axis=1)
    
    #std_col_df = pd.DataFrame(predictions_df.std(axis=1), index=X_test.index, columns=["std"])
    #predictions_df = pd.concat([predictions_df, std_col_df], axis=1)
    
    static_predictions_df = pd.DataFrame(static_predictions, index=X_test.index, columns=["predicted"])

    examine_df_part = pd.concat([static_predictions_df, y_test], axis=1)
    examine_df = pd.concat([examine_df_part, predictions_df], axis=1)

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

#!/bin/python3

import argparse
import logging
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import utility


def predict_lr(input_file, output_path, document_set):
    logging.info("Logistic regression model requested on %s...", input_file)

    logging.info("Loading document set...")
    #with open(document_set, 'r') as infile:
        #docs = json.load(infile)
    docs = pd.read_json(document_set)

    logging.info("Loading features...")
    with open(input_file, 'r') as infile:
        features = json.load(infile)

    X = features
    y = docs.source

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    clf = LogisticRegression(random_state=42, multi_class='ovr')
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    print(score)
    
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
        help="The document set",
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    ARGS = parse()
    utility.init_logging(ARGS.log_path)
    input_path, output_path = utility.fix_paths(ARGS.experiment_path, ARGS.input_path, ARGS.output_path)
    documents_path, output_path = utility.fix_paths(ARGS.experiment_path, ARGS.documents, ARGS.output_path)

    predict_lr(input_path, output_path, documents_path)

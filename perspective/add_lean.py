#!/bin/python3

import argparse
import logging
import json
import os
import pandas as pd

import utility


def add_lean(input_file, output_file, bias_file):
    logging.info("Leaning table requested on %s...", input_file)

    with open(bias_file, 'r') as infile:
        bias = json.load(infile)

    # add a new column
    logging.info("Loading documents...")
    with open(input_file, 'r') as infile:
        docs = json.load(infile)

    leanings = pd.Series()
    for doc in docs:
        lean = 'center'
        source = doc["source"]
        if bias[source] < -.666:
            lean = 'left'
        elif bias[source] > .666:
            lean = 'right'

        doc["lean"] = lean
    
    # write out the file
    logging.info("Saving document data to '%s'", output_path)
    with open(output_path, 'w') as outfile:
        json.dump(docs, outfile)

    logging.info("document data saved to '%s'", output_path)

def parse():
    """Handle all command line argument parsing.

    Returns the parsed args object from the parser
    """
    parser = argparse.ArgumentParser()
    parser = utility.add_common_parsing(parser)

    parser.add_argument(
        "--bias-file",
        dest="bias_file",
        type=str,
        required=False,
        default="",
        metavar="<str>",
        help="the csv with the leanings",
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    ARGS = parse()
    utility.init_logging(ARGS.log_path)
    input_path, output_path = utility.fix_paths(ARGS.experiment_path, ARGS.input_path, ARGS.output_path)
    bias_path, output_path = utility.fix_paths(ARGS.experiment_path, ARGS.bias_file, ARGS.output_path)

    add_lean(input_path, output_path, bias_path)

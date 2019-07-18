#!/bin/python3

"""Combines two sets of features (such as TFIDF and AS-VEC)"""

import argparse
import logging
import json
import os
import pandas as pd

import utility


def combine(input_file_1, input_file_2, output_file, overwrite=False):
    logging.info("Combined feature set requested from %s and %s", input_file_1, input_file_2)

    if not utility.check_output_necessary(output_file, overwrite):
        return

    df1 = pd.read_json(input_file_1)
    df2 = pd.read_json(input_file_2)

    combined_df = pd.concat([df1, df2], axis=1)

    logging.info("Saving combined feature set to '%s'", output_file)

    rows = combined_df.values.tolist()
    with open(output_file, 'w') as outfile:
        json.dump(rows, outfile)

    logging.info("Feature set saved")

def parse():
    """Handle all command line argument parsing.

    Returns the parsed args object from the parser
    """
    parser = argparse.ArgumentParser()
    parser = utility.add_common_parsing(parser)

    parser.add_argument(
        "--input-2",
        dest="input_2",
        type=str,
        required=False,
        default=None,
        metavar="<str>",
        help="The second input file",
    )

    cmd_args = parser.parse_args()
    return cmd_args


if __name__ == "__main__":
    ARGS = parse()
    utility.init_logging(ARGS.log_path)
    input_path, output_path = utility.fix_paths(ARGS.experiment_path, ARGS.input_path, ARGS.output_path)
    input_path2, output_path = utility.fix_paths(ARGS.experiment_path, ARGS.input_2, ARGS.output_path)

    combine(input_path, input_path2, output_path, overwrite=ARGS.overwrite)

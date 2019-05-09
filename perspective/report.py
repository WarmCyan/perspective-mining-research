#!/usr/bin/env python

"""Compiles statistics on dataset, TFIDF, and AS vectors."""

import argparse
import datetime
import logging
import pandas as pd

import utility


def gen_report(input_folder, output_path, documents=None, tfidf=None, aspects=None, aspect_sentiments=None, overwrite=False):
    
    if documents is not None:
        report_documents(documents, output_path)

    if tfidf is not None:
        pass

    if aspects is not None:
        pass

    if aspect_sentiments is not None:
        pass

def report_documents(path, output_path):
    logging.info("Creating documents report...")
    df = pd.read_json(path)

    with open(output_path + "/report_documents.txt", 'w') as out_file:
        out_file.write("Documents report generated on " + str(datetime.datetime.now()))
        out_file.write("\n===================================\n")

        out_file.write("\n\nNumber of articles: " + str(df.shape[0]) + "\n\n")
        out_file.write("Sources:\n")
        out_file.write(str(df.source.value_counts()) + "\n")

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
        default=None,
        metavar="<str>",
        help="The input path for the documents json",
    )
    
    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    ARGS = parse()
    utility.init_logging(ARGS.log_path)
    input_path, output_path = utility.fix_paths(ARGS.experiment_path, ARGS.input_path, ARGS.output_path)
    documents_path, output_path = utility.fix_paths(ARGS.experiment_path, ARGS.documents, ARGS.output_path)


    gen_report(input_path, output_path, documents_path, overwrite=ARGS.overwrite)

#!/usr/bin/env python

"""Prune article set to contain only those that meet a minimum threshold of words
and from sources that meet a minimum threshold of articles.

Primarily intended for webhose set.
"""

import argparse
import os.path
import logging
import pandas as pd

import utility

def prune_articles(input_file, output_file, source_threshold=20, word_threshold=500, overwrite=False):
    logging.info("Pruned article set requested for webhose dataset at '%s', minimum source count of " +
            "%i and minimum word count of %i", output_file, source_threshold, word_threshold)

    # check to see if the output path already exists
    if not utility.check_output_necessary(output_file, overwrite):
        return

    # load the data
    article_table = pd.read_csv(input_file)
    
    # prune by word count
    article_table = article_table[article_table.apply(lambda x: x['text'].count(' ') > word_threshold, axis=1)]

    # prune by source article count
    source_counts = article_table.site.value_counts()
    acceptable_sources = list(source_counts[source_counts > source_threshold].index)
    article_table = article_table[article_table.site.isin(acceptable_sources)]

    logging.info("Pruned to %i articles", article_table.shape[0])
    
    # make the output path if it doens't exist
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    # write it out
    article_table.to_csv(output_file)

def parse():
    """Handle all command line argument parsing.

    Returns the parsed args object from the parser
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        type=str,
        required=True,
        metavar="<str>",
        help="The name and path for the output sentence data",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        metavar="<str>",
        help="The path to csv of raw input data",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Specify this flag to overwrite existing output data if they exist",
    )
    parser.add_argument(
        "--thresh_source",
        dest="thresh_source",
        type=int,
        required=False,
        default=20,
        metavar="<int>",
        help="The minimum number/threshold of articles needed from a source",
    )
    parser.add_argument(
        "--thresh_words",
        dest="thresh_words",
        type=int,
        required=False,
        default=500,
        metavar="<int>",
        help="The minimum number/threshold of words needed in an article",
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    ARGS = parse()
    prune_articles(ARGS.input_file, ARGS.output_file, ARGS.thresh_source, ARGS.thresh_words, ARGS.overwrite)

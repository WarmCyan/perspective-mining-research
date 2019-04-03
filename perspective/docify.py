#!/usr/bin/env python

"""Creates a documents json from kaggle1 data."""

# NOTE: this script is designed for the kaggle1 dataset
# (it's specific preprocessing for that dataset, so it should probably go somewhere else

import argparse
import os
import logging
import json
from tqdm import tqdm

import pandas as pd

import utility


def docify(input_folder, output_path, count=-1, content_column="content", overwrite=False):
    """Create a file of documents from all csv files in a folder

    A count of -1 means output _all_ documents.
    Input_folder assumes no trailling /
    """

    logging.info("document data requested for '%s' dataset at '%s'...", input_folder, output_path)

    # check to see if the output path already exists
    if not utility.check_output_necessary(output_path, overwrite):
        return

    # load the data
    logging.info("Loading article data...")
    article_table = None

    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".csv"):
            logging.debug("Loading '%s'...", filename)
            article_table_in = pd.read_csv(input_folder + "/" + filename)
            if article_table is None:
                article_table = article_table_in
            else:
                article_table = pd.concat([article_table, article_table_in])
    
    
    #logging.debug("Loading articles1.csv...")
    #article_table1 = pd.read_csv(input_folder + "/articles1.csv")
    #logging.debug("Loading articles2.csv...")
    #article_table2 = pd.read_csv(input_folder + "/articles2.csv")
    #logging.debug("Loading articles3.csv...")
    #article_table3 = pd.read_csv(input_folder + "/articles3.csv")

    #article_table = pd.concat([article_table1, article_table2, article_table3])

    # split every sentence from every article on '.'
    logging.info("Grabbing articles...")
    documents = []
    for article in tqdm(article_table[content_column]):
        if count != -1 and len(documents) > count:
            break

        documents.append(article)


    # write out the file
    logging.info("Saving document data to '%s'", output_path)
    with open(output_path, 'w') as outfile:
        json.dump(documents, outfile)

    logging.info("document data saved to '%s'", output_path)


def parse():
    """Handle all command line argument parsing.

    Returns the parsed args object from the parser
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=str,
        required=True,
        metavar="<str>",
        help="The name and path for the output document data",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_folder",
        type=str,
        required=True,
        metavar="<str>",
        help="The path to the folder containing the kaggle1 raw input data",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Specify this flag to overwrite existing output data if they exist",
    )
    parser.add_argument(
        "-c",
        "--count",
        dest="count",
        type=int,
        required=False,
        default=-1,
        metavar="<int>",
        help="The number of sentences to use",
    )
    parser.add_argument(
        "--col",
        dest="column",
        type=str,
        required=False,
        default="content",
        metavar="<str>",
        help="The name of the column with the article text",
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    ARGS = parse()
    docify(ARGS.input_folder, ARGS.output_path, ARGS.count, ARGS.column, ARGS.overwrite)

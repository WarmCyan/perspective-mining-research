#!/usr/bin/env python

"""Creates a documents json from kaggle1 data."""

# NOTE: this script is designed for the kaggle1 dataset
# (it's specific preprocessing for that dataset, so it should probably go somewhere else

import argparse
import random
import os
import logging
import json
from tqdm import tqdm

import pandas as pd

import utility


def docify(input_folder, output_path, count=-1, content_column="content", source_column="publication", keywords=[], overwrite=False):
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

    # target documents to keywords if supplied
    if len(keywords) > 0 and keywords[0] != "":
        tmp_table = pd.DataFrame(columns=article_table.columns)
        logging.info("Targeting document set to keywords %s...", str(keywords))
        for word in keywords:
            tmp_table = pd.concat([tmp_table, article_table[article_table[content_column].str.contains(word)]])

        tmp_table = tmp_table.drop_duplicates()
        article_table = tmp_table.copy()
        
    # randomly shuffling
    if count != -1:
        logging.info("Shuffling %i subset of documents for output...", count)
        article_table = article_table.sample(count, random_state=42)

    # TODO: include source and content in output
    # TODO: randomize if count is less than 0
    # TODO: need to save more input parameters in log

    # split every sentence from every article on '.'
    logging.info("Grabbing articles...")
    documents = []
    for (idx, row) in tqdm(article_table.iterrows()):
        if count != -1 and len(documents) > count:
            break

        documents.append({"text": row.loc[content_column], "source": row.loc[source_column]})

    # this is literally pointless...the subset has already been taken at this point
    #if count != -1:
        #logging.info("Shuffling %i subset of documents for output...", count)
        #random.shuffle(documents)

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
    parser = utility.add_common_parsing(parser)

    parser.add_argument(
        "--col",
        dest="column",
        type=str,
        required=False,
        default="content",
        metavar="<str>",
        help="The name of the column with the article text",
    )
    
    parser.add_argument(
        "--source-column",
        dest="source_column",
        type=str,
        required=False,
        default="publication",
        metavar="<str>",
        help="The name of the column with the article source",
    )
    
    parser.add_argument(
        "--keywords",
        dest="keywords",
        type=str,
        required=False,
        default="",
        metavar="<str>",
        help="The set of words a document must contain",
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    ARGS = parse()
    utility.init_logging(ARGS.log_path)
    input_path, output_path = utility.fix_paths(ARGS.experiment_path, ARGS.input_path, ARGS.output_path)

    keywords = ARGS.keywords.split(",")


    docify(input_path, output_path, ARGS.count, ARGS.column, ARGS.source_column, keywords, ARGS.overwrite)

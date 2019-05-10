#!/usr/bin/env python

"""Creates sentences for Word2Vec out of an input csv file."""

# NOTE: this script is designed for the kaggle1 dataset
# (it's specific preprocessing for that dataset, so it should probably go somewhere else

import argparse
import os.path
import logging

import pandas as pd

import utility


def sentencify(input_folder, output_path, count=-1, overwrite=False):
    """Create a file of sentences from the 3 csv files.

    A count of -1 means output _all_ sentences.
    Input_folder assumes no trailling /
    """

    logging.info("Sentence data requested for kaggle1 dataset at '%s'...", output_path)

    # check to see if the output path already exists
    if not utility.check_output_necessary(output_path, overwrite):
        return

    # load the data
    logging.info("Loading article data...")
    logging.debug("Loading articles1.csv...")
    article_table1 = pd.read_csv(input_folder + "/articles1.csv")
    logging.debug("Loading articles2.csv...")
    article_table2 = pd.read_csv(input_folder + "/articles2.csv")
    logging.debug("Loading articles3.csv...")
    article_table3 = pd.read_csv(input_folder + "/articles3.csv")

    article_table = pd.concat([article_table1, article_table2, article_table3])

    # split every sentence from every article on '.'
    logging.info("Splitting articles...")
    sentences = []
    for article in article_table.content:
        if count != -1 and len(sentences) > count:
            break

        sentences.extend(article.split("."))

    # chop down to size as needed
    if count != -1 and len(sentences) > count:
        sentences = sentences[:count]

    # write out the file
    logging.info("Saving sentence data to '%s'", output_path)
    with open(output_path, 'w') as file_out:
        for sentence in sentences:
            file_out.write("{0}\n".format(sentence))

    logging.info("Sentence data saved to '%s'", output_path)


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
    input_path, output_path = utility.fix_paths(ARGS.experiment_path, ARGS.input_path, ARGS.output_path)
    
    sentencify(input_path, output_path, ARGS.count, ARGS.overwrite)

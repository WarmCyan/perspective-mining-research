#!/usr/bin/env python

"""Runs kaggle1 dataset through preprocessing steps."""

# NOTE: this is only for kaggle1, this should probably go somewhere else eventually

import argparse
import os.path
import logging

import codecs
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from progress.bar import IncrementalBar

import utility


def parse_doc(doc):
    """Tokenizes, lemmatizes, strips stop words, etc."""
    lmtzr = WordNetLemmatizer()
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(doc.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return text_stem


def preprocess(input_folder, output_path, count=-1, overwrite=False):
    """Run the preprocess process on all documents in dataset."""

    logging.info("Preprocessing requested for kaggle1 dataset at '%s'", output_path)

    # check if output file already exists
    if not utility.check_output_necessary(output_path, overwrite):
        return

    # ensure nltk datasets are present
    logging.debug("Ensuring nltk sets...")
    nltk.download("stopwords")
    nltk.download("wordnet")

    # load the data
    logging.info("Loading article data...")
    logging.debug("Loading articles1.csv...")
    article_table1 = pd.read_csv(input_folder + "/articles1.csv")
    logging.debug("Loading articles2.csv...")
    article_table2 = pd.read_csv(input_folder + "/articles2.csv")
    logging.debug("Loading articles3.csv...")
    article_table3 = pd.read_csv(input_folder + "/articles3.csv")

    article_table = pd.concat([article_table1, article_table2, article_table3])

    out = codecs.open(output_path, 'w', 'utf-8')

    logging.info("Parsing...")
    bar = IncrementalBar("Parsing", max=len(article_table.index))
    counter = 0
    for article in article_table.content:
        if count >= 0 and counter == count:
            break
        counter += 1
        
        tokens = parse_doc(article)
        if len(tokens) > 0:
            out.write(' '.join(tokens) + '\n')
        out.write(" \n")
        bar.next()
    bar.finish()

    logging.info("Preprocessing completed, output at '%s'", output_path)


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
        help="The name and path for the output sentence data",
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
        required=True,
        default=-1,
        metavar="<int>",
        help="The number of articles to handle",
    )

    cmd_args = parser.parse_args()
    return cmd_args


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    ARGS = parse()
    preprocess(ARGS.input_folder, ARGS.output_path, ARGS.count, ARGS.overwrite)

#!/usr/bin/env python

# code drawn from https://github.com/harpaj/Unsupervised-Aspect-Extraction

"""This script takes a file of input data and outputs a word2vec model."""

import argparse
import codecs
import logging
import os.path

import gensim


class Sentences:
    """Class with an iterator for running through every line in an input file."""

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, "r", "utf-8"):
            yield line.split()


def parse():
    """Handle all command line argument parsing.

    Returns the parsed args object from the parser
    """
    parser = argparse.ArgumentParser()

    # input and output files
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=str,
        required=True,
        metavar="<str>",
        help="The name and path for the output word2vec model",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        type=str,
        required=True,
        metavar="<str>",
        help="The path to the file of input data",
    )

    # model parameters
    parser.add_argument(
        "--dim",
        dest="embed_dim",
        type=int,
        metavar="<int>",
        required=False,
        default=200,
        help="Embeddings dimension (default=200)",
    )
    parser.add_argument(
        "--win",
        dest="window",
        type=int,
        metavar="<int>",
        required=False,
        default=5,
        help="Window for word2vec (default=5)",
    )
    parser.add_argument(
        "--min",
        dest="min_count",
        type=int,
        metavar="<int>",
        required=False,
        default=10,
        help="min_count for word2vec (default=10)",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        metavar="<int>",
        required=False,
        default=4,
        help="Number of threads (default=4)",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        metavar="<int>",
        required=False,
        default=2,
        help="Number of epochs (default=2)",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Specify this flag to overwrite existing output models if they exist",
    )

    args = parser.parse_args()
    return args


def main():
    """The main code that gets run on executing this file."""

    args = parse()

    logging.info(
        "Word2vec model requested for input %s, output %s", args.input_path, args.output_path
    )

    # check to see if the output path already exists
    logging.debug("Checking for existence of %s...", args.output_path)
    if os.path.isfile(args.output_path):
        logging.debug("Output model file already exists.")

        # check if should overwite the existing output or not
        if args.overwrite:
            logging.debug("Overwrite requested, continuing...")
            logging.warning("Overwriting an existing model output!")
        else:
            logging.debug("No overwrite requested, skipping word2vec model building...")
            exit()

    logging.info("Running word2vec on %s, outputting to %s...", args.input_path, args.output_path)

    # convert input
    logging.debug("Converting input into sentences...")
    sentences = Sentences(args.input_path)

    # run the model creation function
    logging.debug("Generating model...")
    logging.info(
        "Model params: %i, %i, %i, %i, %i",
        args.embed_dim,
        args.window,
        args.min_count,
        args.workers,
        args.epochs,
    )
    model = gensim.models.Word2Vec(
        sentences,
        size=args.embed_dim,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=1,
        iter=args.epochs,
    )
    model.save(args.output_path)

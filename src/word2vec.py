#!/usr/bin/env python

# code drawn in part from https://github.com/harpaj/Unsupervised-Aspect-Extraction

"""This script takes a file of input data and outputs a word2vec model."""

import argparse
import codecs
import logging
import os.path

import gensim

import utility


class Sentences:
    """Class with an iterator for running through every line in an input file."""

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, "r", "utf-8"):
            yield line.split()


def run(
    input_path,
    output_path,
    embed_dim=200,
    window=5,
    min_count=10,
    workers=4,
    epochs=2,
    overwrite=False,
):
    """The primary function for generating the word2vec model."""

    logging.info("Word2vec model requested for input '%s', output '%s'", input_path, output_path)

    # check to see if the output path already exists
    if not utility.check_output_necessary(output_path, overwrite):
        return

    logging.info("Running word2vec on '%s', outputting to '%s'...", input_path, output_path)

    # convert input
    logging.debug("Converting input into sentences...")
    sentences = Sentences(input_path)

    # run the model creation function
    logging.debug("Generating model...")
    logging.info("Model params: %i, %i, %i, %i, %i", embed_dim, window, min_count, workers, epochs)
    model = gensim.models.Word2Vec(
        sentences,
        size=embed_dim,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        iter=epochs,
    )
    model.save(output_path)

    logging.info("Word2Vec model saved to '%s'", output_path)


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

    cmd_args = parser.parse_args()
    return cmd_args


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    ARGS = parse()
    run(
        ARGS.input_path,
        ARGS.output_path,
        ARGS.embed_dim,
        ARGS.window,
        ARGS.min_count,
        ARGS.workers,
        ARGS.epochs,
        ARGS.overwrite,
    )

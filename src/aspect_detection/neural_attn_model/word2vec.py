#!/usr/bin/env python

# code drawn from https://github.com/harpaj/Unsupervised-Aspect-Extraction

"""This script takes a file of input data and outputs a word2vec model."""

import argparse
import codecs
import logging

import gensim


class Sentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, "r", "utf-8"):
            yield line.split()


# Parse commandline arguments

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

sentences = Sentences(args.input_path)
model = gensim.models.Word2Vec(sentences, size=args.embed_dim, window=args.window, min_count=args.min_count, workers=args.workers, sg=1, iter=args.epochs)
model.save(args.output_path)

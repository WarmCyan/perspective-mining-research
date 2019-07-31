#!/bin/python3
import logging
import json
import argparse
import pandas as pd

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

import utility


# NOTE: expects the documents, not the tokenized
# if you want to use same vocab as aspects, pass the path to match_vocab (expects dictionary where keys are vocab)
def tfidf(input_path, output_path, feature_count=5000, match_vocab=None, overwrite=False):

    logging.info("TF-IDF requested on document set '%s'...", input_path)

    if not utility.check_output_necessary(output_path, overwrite):
        return

    logging.info("Loading documents...")

    with open(input_path, 'r') as infile:
        docs = json.load(infile)

    corpus = []
    for document in tqdm(docs):
        corpus.append(document["text"])

    vocab = None
    if match_vocab:
        with open(match_vocab, 'r') as in_file:
            vocab_data = json.load(in_file)
            vocab = list(set([word.lower() for word in vocab_data.keys()]))
            print(vocab)

    logging.info("Running TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=feature_count, vocabulary=vocab)
    vectorizer.fit(corpus)
    tfidf_matrix = vectorizer.transform(corpus)
    #tfidf_matrix = vectorizer.fit_transform(corpus)

    logging.info("Saving TF-IDF matrix...")
    with open(output_path, 'w') as outfile:
        json.dump(tfidf_matrix.todense().tolist(), outfile)
    #df = pd.DataFrame.from_records(tfidf_matrix.todense())
    ##df.columns = vectorizer.get_feature_names()
    
    #df.to_json(output_path)
    
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

    tfidf(input_path, output_path, ARGS.overwrite)

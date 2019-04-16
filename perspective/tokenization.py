#!/bin/python3

import nltk
from nltk.corpus import stopwords
import argparse
import logging
import json
import os

from tqdm import tqdm
import utility

def tokenize(input_file, output_path, count=-1, overwrite=False):
    logging.info("Tokenization requested on document set '%s'...", input_file)

    if not utility.check_output_necessary(output_path + "/pos.json", overwrite):
        return
    
    with open(input_file, 'r') as infile:
        docs = json.load(infile)

    if count > 0:
        logging.info("Selecting document subset of size %i", count)
        docs = docs[0:count]
        
    # make the output path if it doens't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sentences = []
    pos_sentences = []

    document_sentences = []
    sentence_documents = []

    logging.info("Sentencifying documents...")
    sentence_index_start = 0
    doc_index = 0
    for doc in tqdm(docs):

        # tokenize
        local_sentences = nltk.sent_tokenize(doc)
        count = len(local_sentences)

        # add the associated sentence id's to the document sentences
        document_sentences.append(list(range(sentence_index_start, sentence_index_start + count)))
        sentence_index_start += count

        # add the associated document id to the sentence_documents list
        sentence_documents.extend([doc_index]*count)
        doc_index += 1

        # add the tokenized sentences
        sentences.extend(local_sentences)

    logging.info("Tokenizing sentences...")
    for sentence in tqdm(sentences):
        # pos tagger
        words = nltk.word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalpha() and word is not "s"]
        tagged = nltk.pos_tag(words)
        pos_sentences.append(tagged)

    #return pos_sentences, sentences, document_sentences, sentence_documents
    
    pos_path = output_path + "/pos.json"
    doc_sent_path = output_path + "/doc_sent.json"
    sent_doc_path = output_path + "/sent_doc.json"
    logging.info("Saving tokenization information...")
    with open(pos_path, 'w') as file_out:
        json.dump(pos_sentences, file_out)
    with open(doc_sent_path, 'w') as file_out:
        json.dump(document_sentences, file_out)
    with open(sent_doc_path, 'w') as file_out:
        json.dump(sentence_documents, file_out)


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

    tokenize(input_path, output_path, ARGS.count, ARGS.overwrite)

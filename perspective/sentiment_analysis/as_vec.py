#!/bin/python3

import json
import nltk
import logging
import argparse

from nltk.corpus import sentiwordnet as swn
from tqdm import tqdm

import os, sys
sys.path.insert(0, os.path.abspath("../../"))
from perspective import utility
#from perspective import utility

# NOTE: expecting an aspects.json, pos.json, sent_doc.json, doc_sent.json
def create_as_vectors(input_path, output_path, minimum_flr=10.0, overwrite=False):
    logging.info("Aspect-sentiment vectors requested for collection at '%s'...", input_path)

    nltk.download('sentiwordnet')

    if not utility.check_output_necessary(output_path, overwrite):
        return

    # load data from input_path
    aspect_data = {}
    logging.info("Loading aspects...")
    with open(input_path + "/aspects.json") as in_file:
        aspect_data = json.load(in_file)

    pos_sentences = []
    logging.info("Loading pos sentences...")
    with open(input_path + "/pos.json") as in_file:
        pos_sentences = json.load(in_file)

    logging.info("Loading sentence document associations...")
    with open(input_path + "/sent_doc.json") as in_file:
        sentence_documents = json.load(in_file)

    logging.info("Loading document sentence associations...")
    with open(input_path + "/doc_sent.json") as in_file:
        document_sentences = json.load(in_file)

    # TODO: this needs to be moved elsewhere
    pruned_data = {}
    for aspect in aspect_data.keys():
        if aspect_data[aspect]["flr"] > minimum_flr:
            pruned_data[aspect] = aspect_data[aspect]

    # NOTE: pruned data is subset of aspect_data 
    
    doc_as_vectors = []
    num_docs = len(document_sentences)

    # initialize vectors
    logging.info("Initializing document aspect-sentiment vectors...")
    for i in tqdm(range(0, num_docs)):
        doc_as_vectors.append([])

        for j in range(0, len(pruned_data.keys())):
            doc_as_vectors[i].append(0.0)

            

    # find sentiments
    as_index = 0
    logging.info("Finding sentiment words...")
    for aspect in tqdm(pruned_data.keys()):
        pos_aspect = pruned_data[aspect]["pos"]

        # add a key to the data dictionary to record sentiments
        pruned_data[aspect]["sentiments"] = {}

        # iterate through every sentence mentioning this aspect
        for sentence_index in pruned_data[aspect]["sentences"]:
            sentence = pos_sentences[sentence_index]

            aspect_score = [0.0, 0.0] # pos = [0], neg = [1]

            # find the index of the aspect
            try:
                aspect_index = sentence.index(pos_aspect[0]) # NOTE: if difference between NN and NNS, this won't catch, so that needs to be dealt with
            except:
                pruned_data[aspect]["sentiments"][sentence_index] = aspect_score
                continue

            # iterate on each side
            adjectives = []
            weights = [] # TODO: weight less the farther away from the word it is
            for i in range(1, 6):
                left = aspect_index - i
                right = aspect_index + len(pos_aspect) + i

                # check for adjectives
                if left >= 0:
                    if sentence[left][1] == "JJ":
                        adjectives.append(sentence[left][0])
                        break

                if right < len(sentence):
                    if sentence[right][1] == "JJ":
                        adjectives.append(sentence[right][0])

            # go through each adjective and get a combined score
            for adjective in adjectives:
                try:
                    score = swn.senti_synset(adjective + ".a.01")
                    aspect_score[0] += score.pos_score()
                    aspect_score[1] += score.neg_score()
                except: continue

            pruned_data[aspect]["sentiments"][sentence_index] = aspect_score

            try:
                sentence_doc = sentence_documents[sentence_index]
            except: 
                print(sentence_index)
                exit()
            doc_as_vectors[sentence_doc][as_index] += aspect_score[0]
            doc_as_vectors[sentence_doc][as_index] -= aspect_score[1]


        as_index += 1



    # for each aspect, go through every document and find all sentences with that aspect
    #doc_as_vectors = []
    #num_docs = len(document_sentences)

    ## initialize vectors
    #logging.info("Initializing document aspect-sentiment vectors...")
    #for i in tqdm(range(0, num_docs)):
    #    doc_as_vectors.append([])

    #    for j in range(0, len(pruned_data.keys())):
    #        doc_as_vectors[i].append(0.0)

        
    #logging.info("Calculating sentiment...")
    #for 

    # (fill the aspect sentiment from left to right)
    #logging.info("Calculating sentiment...")
    #for aspect in tqdm(pruned_data.keys()):
    #    aspect_meta = pruned_data[aspect]

    #    # iterate each document
    #    for i in range(0, num_docs):
    #        aspect_score = 0.0

    #        # iterate each sentence of the doc, and if this aspect included, aggregate sentiment
    #        for sentence_index in document_sentences[i]:
    #            if sentence_index in aspect_meta['sentences']:
    #                sentiment = aspect_meta['sentiments'][sentence_index]
    #                aspect_score += sentiment[0]
    #                aspect_score -= sentiment[1]

    #        doc_as_vectors[i].append(aspect_score)

    # make the output path if it doens't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    with open(output_path + "/doc_as_vectors.json", "w") as file_out:
        json.dump(doc_as_vectors, file_out)
    
            
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
        help="The path to the folder for the output data",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        type=str,
        required=True,
        metavar="<str>",
        help="The path to the folder containing aspect and document info json data",
    )
    parser.add_argument(
        "-f",
        "--flr",
        dest="flr",
        type=float,
        required=False,
        default=10.0,
        metavar="<float>",
        help="The minimum flr of an aspect",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Specify this flag to overwrite existing output data if they exist",
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    ARGS = parse()
    create_as_vectors(ARGS.input_path, ARGS.output_path, ARGS.flr, ARGS.overwrite)

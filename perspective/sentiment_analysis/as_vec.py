#!/bin/python3

import json
import nltk

from nltk.corpus import sentiwordnet as swn
from tqdm import tqdm

import os, sys
from perspective import utility
#sys.path.insert(0, os.path.abspath("../../"))
#from perspective import utility

# NOTE: expecting an aspects.json, pos.json, sent_doc.json, doc_sent.json
def create_as_vectors(input_path, output_path, overwrite=False):
    logging.info("Aspect-sentiment vectors requested for collection at '%s'...", input_path)

    nltk.download('sentiwordnet')

    if not utility.check_output_necessary(output_path, overwrite):
        return

    # load data from input_path
    aspect_data = {}
    logging.info("Loading aspects...")
    with open("../data/cache/kaggle_aspects/aspects.json") as in_file:
        aspect_data = json.load(in_file)

    pos_sentences = []
    logging.info("Loading pos sentences...")
    with open("../data/cache/kaggle_aspects/pos.json") as in_file:
        pos_sentences = json.load(in_file)

    document_sentences = []
    logging.info("Loading sentence document associations...")
    with open("../data/cache/kaggle_aspects/sent_doc.json") as in_file:
        sentence_documents = json.load(in_file)

    sentence_documents = []
    logging.info("Loading document sentence associations...")
    with open("../data/cache/kaggle_aspects/doc_sent.json") as in_file:
        document_sentences = json.load(in_file)

    # TODO: this needs to be moved elsewhere
    pruned_data = {}
    for aspect in aspect_data.keys():
        if aspect_data[aspect]["flr"] > 10.0:
            pruned_data[aspect] = aspect_data[aspect]

    # find sentiments
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



    # for each aspect, go through every document and find all sentences with that aspect
    doc_as_vectors = []
    num_docs = len(document_sentences.keys())

    # initialize vectors
    for i in range(0, num_docs):
        doc_as_vectors.append([])

    # (fill the aspect sentiment from left to right)
    for aspect in tqdm(pruned_data.keys()):
        aspect_meta = pruned_data[aspect]

        # iterate each document
        for i in range(0, num_docs):
            aspect_score = 0.0

            # iterate each sentence of the doc, and if this aspect included, aggregate sentiment
            for sentence_index in document_sentences[i]:
                if sentence_index in aspect_meta['sentences']:
                    sentiment = aspect_meta['sentiments'][sentence_index]
                    aspect_score += sentiment[0]
                    aspect_score -= sentiment[1]

            doc_as_vectors[i].append(aspect_score)

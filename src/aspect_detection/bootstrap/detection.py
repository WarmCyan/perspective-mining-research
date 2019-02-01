#!/bin/python3
import nltk
import logging
import itertools
import json
import argparse

from tqdm import tqdm

import flr
import util

# NOTE: conceptually coming from "An unsupervised aspect detection model for sentiment analysis of reviews"

aspect_data = {}


def detect(input_file, count=-1, overwrite=False):
    global aspect_data
    logging.info("Aspect detection requested on document set '%s'...", input_file)

    with open(input_file, 'r') as infile:
        docs = json.load(infile)

    if count > 0: docs = docs[0:count]

    pos_sentences, sentences = tokenize(docs)
    generate_candidates(pos_sentences)
    compute_flr(pos_sentences)

    #print(aspect_data)

    # testing just what's top and what isn't
    sorted_aspects = sorted(aspect_data, key = lambda x: aspect_data[x]["flr"])
    for thing in sorted_aspects:
        yep = aspect_data[thing]
        print(yep["pos"], yep["flr"])


# take in a list of documents, and turn into POS sentences
def tokenize(docs):
    sentences = []
    pos_sentences = []

    logging.info("Sentencifying documents...")
    for doc in tqdm(docs):
        sentences.extend(nltk.sent_tokenize(doc))

    logging.info("Tokenizing sentences...")
    for sentence in tqdm(sentences):
        # pos tagger
        words = nltk.word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalpha() and word is not "s"]
        tagged = nltk.pos_tag(words)
        pos_sentences.append(tagged)

    return pos_sentences, sentences

# TODO: definitely move out
def generate_patterns():
    patterns = []

    # adjective noun combinations "JJ NN", "JJ NN NNS", etc.
    for i in range(1, 4):
        for pattern in itertools.product(["NN", "NNS"], repeat=i):
            build = ["JJ"]
            build.extend(pattern)
            patterns.append(build)

    patterns.append(["DT", "JJ"])
    
    for i in range(1, 3):
        for pattern in itertools.product(["NN", "NNS", "VBG"], repeat=i):
            build = ["DT"]
            build.extend(pattern)
            patterns.append(pattern)
    
    return patterns

def generate_candidates(pos_sentences):
    patterns = generate_patterns()
    logging.info("Generating aspect candidates...")

    index = 0
    for pos_sentence in tqdm(pos_sentences):

        # search for all the patterns
        detect_sentence_aspects(pos_sentence, ["NN", "NNS"], index, False, 1)
        detect_sentence_aspects(pos_sentence, ["NN", "NNS"], index, False, 2)
        detect_sentence_aspects(pos_sentence, ["NN", "NNS"], index, False, 3)
        detect_sentence_aspects(pos_sentence, ["NN", "NNS"], index, False, 4)

        for pattern in patterns:
            detect_sentence_aspects(pos_sentence, pattern, index, True)

        index += 1

# TODO: move out
def detect_sentence_aspects(pos_sentence, pattern, sentence_index, order_matters=True, count=-1):
    global aspect_data

    if order_matters:
        end = len(pos_sentence) - len(pattern)
    else:
        end = len(pos_sentence) - count

    if len(pattern) >= len(pos_sentence) or count >= len(pos_sentence):
        return

    for index, (word, pos) in enumerate(pos_sentence, 1):
        if index == end:
            break

        i = 0
        found = True

        # this is for specific sequences of tags ex: match exactly a pattern of "JJ NN NNS"
        if order_matters:
            # for each word type (in order, enforced by i) in the requested pattern
            for part in pattern:

                # is the word type at point i does not match point i in the pattern, NOPE it out of here
                if part != pos_sentence[index + i][1]:
                    found = False
                    break
                i += 1

        # this is for detecting any combinations of word types in pattern of "count" size
        else:
            # check each word in the sentence from index to maximum pattern size
            for part in pos_sentence[index:index + count]:

                # if the word type isn't anywhere in the pattern, NOPE it out of here
                if part[1] not in pattern:
                    found = False
                    break

        if found:

            aspect = None
            if order_matters:
                aspect = pos_sentence[index:index + len(pattern)]
            else:
                aspect = pos_sentence[index:index + count]
                
            # add to database as needed
            string_aspect = util.stringify_pos(aspect)
            if string_aspect not in aspect_data.keys():
                aspect_data[string_aspect] = {"count": 1, "sentences": [sentence_index], "pos": aspect}
            else:
                aspect_data[string_aspect]["count"] += 1
                aspect_data[string_aspect]["sentences"].append(sentence_index)

def compute_flr(pos_sentences):
    global aspect_data

    logging.info("Computing FLR scores...")

    for aspect in tqdm(aspect_data.keys()):
        aspect_data[aspect]["flr"] = flr.flr(aspect_data[aspect]["pos"], pos_sentences, aspect_data)

def parse():
    """Handle all command line argument parsing.

    Returns the parsed args object from the parser
    """
    parser = argparse.ArgumentParser()

    #parser.add_argument(
    #    "-o",
    #    "--output",
    #    dest="output_path",
    #    type=str,
    #    required=True,
    #    metavar="<str>",
    #    help="The name and path for the output sentence data",
    #)
    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        metavar="<str>",
        help="The path to the json file containing the input documents",
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
        required=False,
        default=-1,
        metavar="<int>",
        help="The number of sentences to use",
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    ARGS = parse()
    detect(ARGS.input_file, ARGS.count, ARGS.overwrite)

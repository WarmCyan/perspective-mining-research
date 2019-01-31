import nltk
import logging
import itertools

from tqdm import tqdm

# NOTE: conceptually coming from "An unsupervised aspect detection model for sentiment analysis of reviews"

aspect_data = {}


def detect(input_file, count=-1, overwrite=False):
    #logging.info("
    pass


# take in a list of documents, and turn into POS sentences
def tokenize(docs):
    sentences = []
    pos_sentences = []

    logging.info("Sentencifying documents...")
    for doc in tqdm(docs):
        sentences.extend(nltk.sent_tokenize(doc))

    logging.info("Tokenizing sentences...")
    for sentence in tqdm(sentences):
        tokenized = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokenized)
        pos_sentences.append(tagged)

    return pos_sentences, sentences

def generate_patterns():
    patterns = []

    # adjective noun combinations "JJ NN", "JJ NN NNS", etc.
    for i in range(1, 4):
        for pattern in itertools.product(["NN", "NNS"], repeat=i):
            build = ["JJ"]
            patterns.append(build.extend(pattern))

    patterns.append("DT", "JJ")
    
    for i in range(1, 3):
        for pattern in itertools.product(["NN", "NNS", "VBG"], repeat=i):
            build = ["DT"]
            patterns.append(build.extend(pattern))
    
    return patterns

def find_aspects(pos_sentences):
    patterns = generate_patterns()
    logging.info("Finding aspects...")

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

def stringify_pos(pos):
    return " ".join([word for word, tag in pos])

def detect_sentence_aspects(pos_sentence, pattern, sentence_index, order_matters=True, count=-1):
    global aspect_data

    if order_matters:
        end = len(pos_sentence) - len(pattern)
    else:
        end = len(pos_sentence) - count

    if len(pattern) >= len(pos_sentence):
        return

    for index, (word, pos) in enumerate(pos_sentence, 1):
        if index == end:
            break

        i = 0
        found = True

        # this is for specific sequences of tags
        if order_matters:
            for part in pattern:
                if part != pos_sentence[index + i][1]:
                    found = False
                    break
                i += 1
        else:
            for part in pos_sentence[index, index + count]:
                if part[1] not in pattern:
                    found = False
                    break

        if found:
            aspect = pos_sentence[index:index + len(pattern)]
            # add to database as needed
            string_aspect = stringify_pos(aspect)
            if string_aspect not in aspect_data.keys():
                aspect_data[string_aspect] = {"count": 1, "sentences": [sentence_index]}
            else:
                aspect_data[string_aspect]["count"] += 1
                aspect_data[string_aspect]["sentences"].append(sentence_index)

"""Functions for computing A-Score metrics."""

import math

def a_score(aspect_data, aspect, num_sentences):
    """Compute the a-score for the given aspect.

    aspect should be a key in aspect_data.
    """

    score = aspect_data[aspect]["count"] * sum_term(aspect_data, aspect, num_sentences)
    return score

def sum_term(aspect_data, aspect, num_sentences):

    term = 0
    for aspect_b in aspect_data.keys():
        log_term = cooccurence_freq(aspect_data, aspect, aspect_b) * num_sentences + 1

        term += math.log(log_term, 2)

    return term

# computes f(a, b_i) / (f(a)*f(b_i))
def cooccurence_freq(aspect_data, aspect1, aspect2):
    aspect1_sentences = aspect_data[aspect1]["sentences"]
    aspect2_sentences = aspect_data[aspect2]["sentences"]

    count = 0

    for sentence_index in aspect1_sentences:
        if sentence_index in aspect2_sentences:
            count += 1

    return count / (len(aspect1_sentences)*len(aspect2_sentences))

"""Functions for computing FLR scores."""

import math

import util

lr_counts = {}

def compute_lr_count(pos_sentences, aspect):
    """For the given aspect, compute the number of left and right word types."""
    
    lr_counts[aspect] = { "l":0, "r":0, "l_seen":[], "r_seen":[] }

    # search every word of every sentence
    for i in range(0, len(pos_sentences)):
        for j in range(1, len(pos_sentences[i])-1):

            # is this the aspect we're looking for?
            if pos_sentences[i][j] == aspect:
                l_type = pos_sentences[i][j-1][1]
                r_type = pos_sentences[i][j+1][1]

                # have we seen the left type before?
                if l_type not in lr_counts[aspect]["l_seen"]:
                    lr_counts[aspect]["l"] += 1
                    lr_counts[aspect]["l_seen"].append(l_type)

                # have we seen the right type before?
                if r_type not in lr_counts[aspect]["r_seen"]:
                    lr_counts[aspect]["r"] += 1
                    lr_counts[aspect]["r_seen"].append(r_type)

def lr_i_calc(pos_sentences, aspect_part, l_index, r_index):
    """Compute the individual LR for a word in an aspect."""
    compute_lr_count(pos_sentences, aspect_part)
    if aspect_part not in lr_counts.keys():
        compute_lr_count(pos_sentences, aspect_part)

    return math.sqrt(lr_counts[aspect_part]["l"]*l_index*lr_counts[aspect_part]["r"]*r_index)

def lr_calc(pos_sentences, aspect):
    """Compute the LR score for an aspect."""
    product = 1

    l_index = 1
    r_index = len(aspect)
    for part in aspect:
        product *= lr_i_calc(pos_sentences, part[0], l_index, r_index)
        l_index += 1
        r_index -= 1 # TODO: ensure you did this right and there's no off-by-one thing

    return product ** (1 / len(aspect))

def flr(aspect, pos_sentences, aspect_data):
    """Compute the FLR score for a given aspect."""
    return aspect_data[util.stringify_pos(aspect)]["count"]*lr_calc(pos_sentences, aspect)

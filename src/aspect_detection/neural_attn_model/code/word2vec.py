#!/usr/bin/env python
#  -*- coding: utf-8  -*-

import gensim
import codecs


class Sentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()


def main(domain):
    source = '../preprocessed_data/%s/train.txt' % domain
    model_file = '../preprocessed_data/%s/w2v_embedding' % domain
    sentences = Sentences(source)
    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=10, workers=4, sg=1, iter=2)
    model.save(model_file)


if __name__ == "__main__":
    print('Pre-training word embeddings ...')
    main('restaurant')
    main('beer')

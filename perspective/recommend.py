import sys
import json
import pandas as pd
import numpy as np
from collections import OrderedDict

from sklearn.metrics.pairwise import cosine_similarity



aspect_data = None
articles = None
as_df = None


def load_data():
    print("Loading data...")
    article_table = pd.read_json("../data/cache/climatetarget_sa_updated/documents.json")
    print("Loaded!")
    return article_table


def load_as_vec(folder):
    print("Loading aspect vector data...")
    pathname = "../data/cache/experiments/vec_" + folder + "/as_vec/doc_as_vectors.json"
    print(pathname)
    as_df = pd.read_json(pathname)
    print("Loaded!")
    print(as_df.shape)
    return as_df


def load_aspects(folder):
    print("Loading aspects...")
    with open("../data/cache/experiments/vec_" + folder + "/aspects/aspects.json", "r") as in_file:
        aspect_data = json.load(in_file)
    print("Loaded!")
    return aspect_data


def oneify(x):
    if x != 0.0:
        return 1.0
    return 0.0
    

def make_presence_df(as_df):
    presence_df = as_df.copy()
    presence_df = presence_df.applymap(oneify)
    return presence_df


def get_article_sentiment(index):
    article = articles.iloc[index].text
    article_as = as_df.iloc[index]

    # find all non-zero aspect sentiments for this article
    sentiment_indices = {}
    for index, val in article_as.iteritems():
        if val != 0.0:
            sentiment_indices[index] = val

    # list all sentiments in the article
    num_aspects = len(list(aspect_data.keys()))

    labeled_sentiments = OrderedDict()

    # iterate each sentiment index
    for key in sentiment_indices.keys():
        neg = False
        obj = False

        aspect_key = key

        # figure out if it's positive, negative, or objective
        if key >= num_aspects and key < num_aspects*2:
            neg = True
            aspect_key = key - num_aspects
        elif key >= num_aspects*2:
            obj = True
            aspect_key = key - num_aspects*2

        # find the actual aspect name
        label = list(aspect_data.keys())[aspect_key]
        if neg: label += "_neg"
        if obj: label += "_obj"
        value = sentiment_indices[key]
        labeled_sentiments[label] = value

    # sort by sentiment value (strength)
    labeled_sentiments = OrderedDict(sorted(labeled_sentiments.items(), key=lambda x:x[1]))

    return sentiment_indices, labeled_sentiments


def print_sentiments(sentiments_dictionary):
    for key, value in sentiments_dictionary:
        print(key, value)


def run(folder, index=0):
    global articles, as_df, aspect_data
    articles = load_data()
    as_df = load_as_vec(folder)
    aspect_data = load_aspects(folder)

    article1 = articles.iloc[index].text
    article1_as = as_df.iloc[index]
    _, article1_sentiments = get_article_sentiment(index)

    # find articles that have similar aspects mentioned
    p_df = make_presence_df(as_df)
    cs_p = cosine_similarity(p_df)
    cs_p_df = pd.DataFrame(cs_p)
    print(cs_p_df)

    sim = cs_p_df.loc[index]
    #closest = sim[(sim < .99) & (sim > .25)].sort_values(ascending=False)
    print(sim.sort_values(ascending=False))
    closest = sim[(sim < .99) & (sim > .25)].sort_values(ascending=False)
    print("Within closeness criteria: ", closest.shape[0])
    print(closest)
    
    closest_indices = list(closest.index)
    closest_indices.insert(0, 0)

    # find the farthest apart sentiment wise of the ones that are close presence wise
    similar_as_df = as_df.iloc[closest_indices]

    as_similarity = pd.DataFrame(cosine_similarity(similar_as_df))
    as_similarity.index = similar_as_df.index
    index2 = as_similarity.sort_values(by=[index]).index[0]

    article2 = articles.iloc[index2].text
    article2_as = as_df.iloc[index2]
    _, article2_sentiments = get_article_sentiment(index2)


    print("=====================================")
    print("Article 1")
    print("=====================================")

    print(article1)
    print("-------------------------------------")
    print_sentiments(article1_sentiments)

    print("=====================================")
    print("Article 2")
    print("=====================================")
    
    print(article2)
    print("-------------------------------------")
    print_sentiments(article2_sentiments)


if __name__ == "__main__":
    
    path = sys.argv[1]
    index = int(sys.argv[2])

    run(path, index)

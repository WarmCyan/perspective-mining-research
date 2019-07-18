#!/bin/python3

import argparse
import logging
import json
import os
import pandas as pd
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

import utility

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    classes = list(set(classes)) # classes[unique_labels(y_true, y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, cm

def predict_lr(input_file, output_path, document_set, undersample=False, oversample=False, name='', model_type="lr", class_balance=False, predict_lean=False, **kwargs):
    logging.info("Logistic regression model requested on %s...", input_file)

    logging.info("Loading document set...")
    #with open(document_set, 'r') as infile:
        #docs = json.load(infile)
    docs = pd.read_json(document_set)

    logging.info("Loading features...")
    with open(input_file, 'r') as infile:
        features = json.load(infile)

    X = features
    y = docs.source

    if predict_lean:
        y = docs.lean

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    if undersample:
        logging.info("Undersampling...")
        rus = RandomUnderSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    if oversample:
        logging.info("Oversampling...")
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    logging.info("Training...")

    if model_type == "lr":
        if class_balance:
            clf = LogisticRegression(random_state=42, multi_class='ovr', class_weight='balanced', max_iter=10000)
        else:
            clf = LogisticRegression(random_state=42, multi_class='ovr', max_iter=10000)
    elif model_type == "mlp":
        #clf = MLPClassifier(hidden_layer_sizes=(200, 100), verbose=True, random_state=42, early_stopping=True, n_iter_no_change=10)
        size = len(X[0])
        clf = MLPClassifier(hidden_layer_sizes=(size), verbose=True, random_state=42, early_stopping=True, n_iter_no_change=10)
    elif model_type == "nb":
        clf = MultinomialNB()
        
    clf.fit(X_train, y_train)

    logging.info("Scoring...")
    score = clf.score(X_test, y_test)
    logging.info("Score: %s", str(score))

    logging.info("Recording confusion matrix...")
    predictions = clf.predict(X_test)

    if predict_lean:
        figure, matrix = plot_confusion_matrix(y_test, predictions, list(docs.lean))
    else:
        figure, matrix = plot_confusion_matrix(y_test, predictions, list(docs.source))
    
    # make the output path if it doens't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    pickle.dump(clf, open(output_path + "/" + model_type + "_" + name + "_model", "wb"))
    
    with open(output_path + "/" + model_type + "_" + name + "_score", 'w') as out_file:
        logging.info("Saving score to " + output_path + "/" + model_type + "_" + name + "_score...")
        out_file.write(str(score))

    results_json_path = output_path + "/results.json"
        
    result_rows = []
    if os.path.exists(results_json_path):
        with open(results_json_path, 'r') as in_file:
            result_rows = json.load(in_file)
    
    result_rows.append({"name":model_type + "_" + name, "score": score, "lean":predict_lean, "balanced":class_balance})

    with open(results_json_path, 'w') as out_file:
        json.dump(result_rows, out_file)

    #
    #if not os.path.exists(results_csv_path):
        #results_df = pd.DataFrame(columns=["name", "score", "lean", "balanced"])
    #else:
        #results_df = pd.read_csv(results_csv_path)
    # results_df.append({"name":model_type + "_" + name, "score": score, "lean":predict_lean, "balanced":class_balance}, ignore_index=True)
    #results_df.loc[results_df.shape[0]] = [model_type + "_" + name, score, predict_lean, class_balance]
    #results_df.loc[len(results_df)] = {"name":model_type + "_" + name, "score": score, "lean":predict_lean, "balanced":class_balance}
    #results_df = results_df.append({"name":model_type + "_" + name, "score": score, "lean":predict_lean, "balanced":class_balance}, ignore_index=True)
    #results_df.to_csv(results_csv_path)

    figure.savefig(output_path + "/" + model_type + "_" + name + "_cm.png")
    
    
def parse():
    """Handle all command line argument parsing.

    Returns the parsed args object from the parser
    """
    parser = argparse.ArgumentParser()
    parser = utility.add_common_parsing(parser)

    parser.add_argument(
        "--documents",
        dest="documents",
        type=str,
        required=False,
        default="",
        metavar="<str>",
        help="the document set",
    )
    parser.add_argument(
        "--name",
        dest="model_name",
        type=str,
        required=False,
        default="",
        metavar="<str>",
        help="the name to save",
    )
    parser.add_argument(
        "--type",
        dest="model_type",
        type=str,
        required=False,
        default="lr",
        metavar="<str>",
        help="the type of model to use",
    )
    parser.add_argument(
        "--undersample",
        dest="undersample",
        action="store_true",
        help="Specify this flag to undersample data",
    )
    parser.add_argument(
        "--oversample",
        dest="oversample",
        action="store_true",
        help="Specify this flag to oversample data",
    )
    parser.add_argument(
        "--balanced",
        dest="balanced",
        action="store_true",
        help="Specify this flag to use scikit learn's balancing argument",
    )
    
    parser.add_argument(
        "--lean",
        dest="lean",
        action="store_true",
        help="Specify this flag to predict lean instead of source",
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    ARGS = parse()
    print("=====NAME:", ARGS.model_name)
    utility.init_logging(ARGS.log_path)
    input_path, output_path = utility.fix_paths(ARGS.experiment_path, ARGS.input_path, ARGS.output_path)
    documents_path, output_path = utility.fix_paths(ARGS.experiment_path, ARGS.documents, ARGS.output_path)

    print("=====NAME:", ARGS.model_name)

    predict_lr(input_path, output_path, documents_path, ARGS.undersample, ARGS.oversample, name=ARGS.model_name, model_type=ARGS.model_type, class_balance=ARGS.balanced, predict_lean=ARGS.lean)

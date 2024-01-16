# FIRST: RENAME THIS FILE TO sentiment_utils.py 

# YOUR NAMES HERE: 


"""
Felix Muzny
CS 4/6120
Homework 4
Fall 2023

Utility functions for HW 4, to be imported into the corresponding notebook(s).

Add any functions to this file that you think will be useful to you in multiple notebooks.
"""
# fancy data structures
from collections import defaultdict, Counter
from pyexpat import XML_PARAM_ENTITY_PARSING_UNLESS_STANDALONE
# for tokenizing and precision, recall, f_measure, and accuracy functions
import nltk
from nltk.metrics.scores import (precision, recall, f_measure, accuracy)
# for plotting
import matplotlib.pyplot as plt
# so that we can indicate a function in a type hint
from typing import Callable
nltk.download('punkt')
import numpy as np
import time

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from typing import Union

from sklearn.feature_extraction.text import CountVectorizer

ModelType = Union[LogisticRegression, GaussianNB, MLPClassifier]

def generate_tuples_from_file(training_file_path: str, delin: str="\t", header: list=["id", "text", "label"], hasheading: bool=False) -> list:
    """
    Generates tuples from file formated like:
    id\ttext\tlabel
    id\ttext\tlabel
    id\ttext\tlabel
    Parameters:
        training_file_path - str path to file to read in
    Return:
        a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    """
    # PROVIDED
    f = open(training_file_path, "r", encoding="utf8")
    skipline = False
    if hasheading:
        skipline = True
    X = []
    y = []
    for review in f:
        if len(review.strip()) == 0:
            continue
        dataInReview = review.strip().split(delin)
        if skipline:
            skipline = False
        elif len(dataInReview) != len(header):
            continue
        elif header[1] == "label":
            y.append(int(dataInReview[1]))
            X.append(nltk.word_tokenize(dataInReview[2]))
        else:
            t = tuple(dataInReview)
            if (not t[2] == '0') and (not t[2] == '1'):
                print("WARNING")
                continue
            X.append(nltk.word_tokenize(t[1]))
            y.append(int(t[2]))
    f.close()  
    # x is tokenized text
    # y is label
    return X, y

"""
NOTE: for all of the following functions, we have prodived the function signature and docstring, *that we used*, as a guide.
You are welcome to implement these functions as they are, change their function signatures as needed, or not use them at all.
Make sure that you properly update any docstrings as needed.
"""

def train_dev_split(data: list, train_ratio: float) -> tuple:
    """
    Split the dataset into train and dev set.
    Args:
        data: list of data in format: [[[word1, word2], [word3, word4], ...], [label1, label2, ...]]
        train_ratio: how many percent of data to be train set, the rest (1-train_ratio) will be dev set
    Returns: 
        tuples of X_train, X_dev, y_train and y_dev
    """

    # get the words and labels
    words_list = data[0]
    labels = data[1]

    # calculate the index that use to split the data
    train_words_size = int(len(words_list) * train_ratio)
    train_labels_size = int(len(labels) * train_ratio)

    # split the data into train and set data
    X_train = words_list[:train_words_size]
    X_dev = words_list[train_words_size:]
    y_train = labels[:train_labels_size]
    y_dev = labels[train_labels_size:]

    return X_train, X_dev, y_train, y_dev

def get_prfa(dev_y: list, preds: list,verbose=False) -> tuple: # 
    """
    Calculate precision, recall, f1, and accuracy for a given set of predictions and labels.
    Args:
        dev_y: list of labels
        preds: list of predictions
        verbose: whether to print the metrics
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    if len(set(preds)) == 1:
        raise Exception("All predictions are the same.")
    precision_ = metrics.precision_score(dev_y, preds)
    recall_ = metrics.recall_score(dev_y, preds)
    f1_ = metrics.f1_score(dev_y, preds)  
    accuracy_ = metrics.accuracy_score(dev_y, preds)

    # whether to print out the result
    if verbose:
        print("Precision:", precision_)
        print("Recall:", recall_)
        print("F1:", f1_)
        print("Accuracy:", accuracy_)

    return precision_, recall_, f1_, accuracy_

def train_and_pf(model: ModelType, train_data: list, dev_data: list, verbose: bool = False):
    """
    Train the chosen model with training_data and get the prediction, then
    using the pred to get the metrics
    Args:
        model: a machine learning model that takes in trainning data and returns prediction
        train_data: a list of training data in the format [(feats, label), ...]
        dev_data: a list of dev data in the format [(feats, label), ...]
        verbose: whether to print the metrics
    Returns:
        tuple of precision, recall, f1, and accuracy 
    """
    # split up the features and labels
    X_train = [data[0] for data in train_data]
    y_train = [data[1] for data in train_data]
    X_dev = [data[0] for data in dev_data]
    y_dev = [data[1] for data in dev_data]

    # train the model
    model.fit(X_train, y_train)

    # get the prediction
    y_pred = model.predict(X_dev)

    # get the metrics
    metrics = get_prfa(y_dev, y_pred, verbose)

    return metrics

# Binary vectorizer
vectorizer = CountVectorizer(input='content', stop_words='english', binary=True)


# Multinomial vectorizer
#vectorizer = CountVectorizer(input='content', stop_words='english', binary=False)

def train(train_feats: list, dev_feats: list, model: ModelType, kind: str, subset: int, verbose: bool = False):
    # This will be the train function
    if "Naive Bayes" in kind: # Change this now to incorporate the data restructuring from the notebook
        start_time = time.time()
        train_subset = []
        for j in range(subset):
            train_subset.append((train_feats[0][j], train_feats[1][j]))
        classifier = model.train(train_subset)
        if verbose:
            print("Took {} s to train Naive Bayes model".format(str(time.time()-start_time)))
        return classifier, None

    elif "Neural" in kind:
        train_subset = vectorizer.transform([' '.join(text) for text in train_feats[0][:subset]]).toarray()
        x_dev = vectorizer.transform([' '.join(text) for text in dev_feats[0]]).toarray()
        y_dev = np.asarray(dev_feats[1])
        model.fit(train_subset, np.asarray(train_feats[1][:subset]), batch_size=64, epochs=3, validation_data=(x_dev,y_dev))
        if verbose:
            model.summary # to see how many trainable parameters
        return model, x_dev

def classify(model: ModelType, dev_feats, kind: str, x_dev = None, verbose: bool = False):
    # Classify/Predict function
    if "Naive Bayes" in kind:
        classification = [model.classify(text) for text in dev_feats[0]] # This will have to restructure dev_feats first to get (dictlist, labellist)
    
    elif "Neural" in kind:
        classification = [round(model.predict(np.asarray([x]), verbose=False)[0][0]) for x in x_dev] # This seems to be necessary for it to classify all 200 data points, verbose=False because it reports on all 200 predictions
        if len([val for val in classification if val == 0]) in [0, len(classification)] and verbose:
            print("All predictions are {}".format(str(classification[0])))
    
    return classification

def _create_training_graph(metrics_fun: Callable, train_feats: list, dev_feats: list, model: ModelType, kind: str, savepath: str = None, verbose: bool = False) -> None:
    """
    Create a graph of the classifier's performance on the dev set as a function of the amount of training data.
    Args:
        metrics_fun: a function that takes in training data and dev data and returns a tuple of metrics
        train_feats: a list of training data in the format [(feats, label), ...]  -> in the format [[train_strings], [train_labels]]
        dev_feats: a list of dev data in the format [(feats, label), ...]
        model: a machine learning model we need to use, should match the "kind"
        kind: the kind of model being used (will go in the title)
        savepath: the path to save the graph to (if None, the graph will not be saved)
        verbose: whether to print the metrics
    """
    #TODO: implement this function
    # create a graph of your classifier's performance on the dev set as a function of the amount of training data
    # the x-axis should be the amount of training data (as a percentage of the total training data)
    # the y-axis should be the performance of the classifier on the dev set
    # the graph should have 4 lines, one for each of precision, recall, f1, and accuracy
    # the graph should have a legend, title, and axis labels
    
    x_vals, f1s, accuracies, precisions, recalls = [], [], [], [], []
    
    for i in range(10,101,10):
        pct = float(i)/float(100)
        if verbose: 
            print('{Percent}% training data'.format(Percent=str(i)))
        #x_vals.append(pct)
        total = len(train_feats[0])
        subset = round(total * pct)
        x_vals.append(subset)
        
        # Train
        model, x_dev = train(train_feats, dev_feats, model, kind, subset, verbose=verbose)
            
        # Predict dev
        classification = classify(model, dev_feats, kind, x_dev = x_dev, verbose=verbose)
            
        reality = dev_feats[1]
        p,r,f,a = metrics_fun(classification, reality, verbose=verbose)
        f1s.append(f)
        accuracies.append(a)
        precisions.append(p)
        recalls.append(r)
        
    # plot the metrics
    plt.plot(x_vals, precisions, label='precision', color='blue')
    plt.plot(x_vals, recalls, label='recall', color='orange')
    plt.plot(x_vals, f1s, label='f1', color='pink')
    plt.plot(x_vals, accuracies, label='accuracy', color='purple')

    plt.legend()
    plt.xlabel("size of training data")
    plt.ylabel('performance')
    plt.title(f"performance of {kind}".format(kind=kind))

    if savepath:
        plt.savefig(savepath)
        plt.show()


def create_index(all_train_data_X: list) -> list:
    """
    Given the training data, create a list of all the words in the training data.
    Args:
        all_train_data_X: a list of all the training data in the format [[word1, word2, ...], ...]
    Returns:
        vocab: a list of all the unique words in the training data
    """
    # figure out what our vocab is and what words correspond to what indices
    #TODO: implement this function
    unique = [list(set(row)) for row in all_train_data_X]

    return unique


def featurize(vocab: list, data_to_be_featurized_X: list, binary: bool = False, verbose: bool = False) -> list:
    """
    Create vectorized BoW representations of the given data.
    Args:
        vocab: a list of words in the vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
        verbose: boolean for whether or not to print out progress
    Returns:
        a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    # using a Counter is essential to having this not take forever
    #TODO: implement this function
    
    # implement this function to featurize your data
    # use nltk.word_tokenize to tokenize the emails
    X = np.zeros((data_to_be_featurized_X.shape[0], len(vocab)))

    for index, row in data_to_be_featurized_X.iterrows():
        row = nltk.word_tokenize(row[0].lower())
        counts = [row.count(word) for word in vocab]

        X[index] = counts

    return X

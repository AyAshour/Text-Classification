import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import nltk
import json
import math


def computeTF(word_dict, doc):
    tf_dict = {}
    doc_words_count = len(doc)
    for word, count in word_dict.items():
        tf_dict[word] = count / float(doc_words_count)
    return tf_dict

def computeIDF(dict, idf_dict):
    N = 1801
    for word, val in dict.items():
        if val > 0:
            idf_dict[word] += 1

    for word, val in idf_dict.items():
        idf_dict[word] = math.log10(N / float(val))

    return idf_dict

def computeTFIDF(tf_doc, idfs):
    tfidf = {}
    for word, val in tf_doc.items():
        tfidf[word] = val*idfs[word]
    return tfidf


def get_distinct_words():
    file = open("file.txt", "r")
    distinct_words = file.read().splitlines()

    return distinct_words


def get_tfidf(review):

    _dict = dict.fromkeys(get_distinct_words(), 0)

    review_tokens = nltk.word_tokenize(review)

    for token in review_tokens:
        if token in _dict:
            _dict[token] += 1

    tf_dict = computeTF(_dict, review_tokens)

    with open('idf_file.txt', 'r') as file:
        loaded_idf_dict = json.loads(file.read())

    idf_dict = computeIDF(_dict, loaded_idf_dict)

    tfidf_dict = computeTFIDF(tf_dict, idf_dict)

    return tfidf_dict

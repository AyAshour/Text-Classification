import os
import nltk
import math
from random import shuffle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import test_data_preprocessing as tdp

positive_files_names = os.listdir("train data/positive")
negative_files_names = os.listdir("train data/negative")

docs_list = []

for file_name in positive_files_names:
    file = open("train data/positive/" + str(file_name), "r")
    docs_list.append(file.read())

for file_name in negative_files_names:
    file = open("train data/negative/" + str(file_name), "r")
    docs_list.append(file.read())

tokenized_docs = []
for doc in docs_list:
    tokens_list = nltk.word_tokenize(doc)
    tokenized_docs.append(tokens_list)


# distinct_words = list(set().union(*tokenized_docs))
distinct_words = tdp.get_distinct_words()


with open('file.txt', 'w') as f:
    for item in distinct_words:
        f.write("%s\n" % item)


dicts_list = []
for i in range(len(docs_list)):
    _dict = dict.fromkeys(distinct_words, 0)
    dicts_list.append(_dict)


for i in range(len(dicts_list)):
    doc = tokenized_docs[i]
    dict = dicts_list[i]
    for word in doc:
        dict[word] += 1

def computeTF(word_dict, doc):
    tf_dict = {}
    doc_words_count = len(doc)
    for word, count in word_dict.items():
        tf_dict[word] = count / float(doc_words_count)
    return tf_dict

tf_dicts_list = []
for i in range(len(dicts_list)):
    doc = tokenized_docs[i]
    dict = dicts_list[i]
    tf_dicts_list.append(computeTF(dict, doc))

def computeIDF(docList):
    N = len(docList)

    # idf_dict = dict.fromkeys(docList[0].keys(), 0)

    # print(idf_dict)
    # for doc in docList:
    #     for word, val in doc.items():
    #         if val > 0:
    #             idf_dict[word] += 1

    # with open('idf_file.txt', 'w') as file:
    #     file.write(json.dumps(idf_dict))

    with open('idf_file.txt', 'r') as file:
        idf_dict = json.loads(file.read())

    for word, val in idf_dict.items():
        idf_dict[word] = math.log10(N/float(val))

    return idf_dict

idfs = computeIDF(dicts_list)

def computeTFIDF(tf_doc, idfs):
    tfidf = {}
    for word, val in tf_doc.items():
        tfidf[word] = val*idfs[word]
    return tfidf

tfidf_dicts_list = []
for tf_doc in tf_dicts_list:
    tfidf_dicts_list.append(computeTFIDF(tf_doc, idfs))

labels_positive = [1] * len(positive_files_names)
labels_negative = [0] * len(negative_files_names)

labels = labels_positive + labels_negative

def shuffle_data(tfidf_dicts, labels):
    temp = list(zip(tfidf_dicts, labels))
    shuffle(temp)
    tfidf_dicts, labels = zip(*temp)

    return tfidf_dicts, labels

tfidf_dicts_list, labels = shuffle_data(tfidf_dicts_list, labels)

tfidf_dicts_list, labels = list(tfidf_dicts_list), list(labels)

tfidf_train_list = []
for dict in tfidf_dicts_list:
    tfidf_train_list.append(list(dict.values()))

tfidf_train_list = np.array(tfidf_train_list)


# Train Model

#
# model = keras.Sequential([
#     keras.layers.Dense(5, input_dim=len(distinct_words), activation=tf.nn.relu),
#     keras.layers.Dense(10, activation=tf.nn.relu),
#     keras.layers.Dense(8, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid),
# ])
#
# model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# model.summary()
#
# checkpoint_path = "model_info/cp.ckpt"
#
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
#
# model.load_weights(checkpoint_path)
# model.fit(tfidf_train_list, labels, epochs=200, callbacks=[cp_callback], batch_size=10)
#
#
# positive_files_names = os.listdir("test data/positive")
# negative_files_names = os.listdir("test data/negative")
#
# docs_list = []
#
# for file_name in positive_files_names:
#     file = open("test data/positive/" + str(file_name), "r")
#     docs_list.append(file.read())
#
# for file_name in negative_files_names:
#     file = open("test data/negative/" + str(file_name), "r")
#     docs_list.append(file.read())
#
# labels_positive = [1] * len(positive_files_names)
# labels_negative = [0] * len(negative_files_names)
#
# labels = labels_positive + labels_negative
#
# tfidf_test_list = []
# for review in docs_list:
#     tfidf_test_list.append(list(tdp.get_tfidf(review).values()))
#
#
# # tfidf_test_list, labels = shuffle_data(tfidf_test_list, labels)
#
#
# test_loss, test_acc = model.evaluate(np.array(tfidf_test_list), labels)
# print('Test accuracy:', test_acc)

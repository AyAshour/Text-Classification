import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Read Data

positive_files_names = os.listdir("dataset/train/positive")
negative_files_names = os.listdir("dataset/train/negative")

docs_list = []

for file_name in positive_files_names:
    file = open("dataset/train/positive/" + str(file_name), "r")
    docs_list.append(file.read())

for file_name in negative_files_names:
    file = open("dataset/train/negative/" + str(file_name), "r")
    docs_list.append(file.read())

labels_positive = [1] * len(positive_files_names)
labels_negative = [0] * len(negative_files_names)

labels = labels_positive + labels_negative
labels = np.array(labels)

MAX_SEQUENCE_LENGTH = 1000

tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs_list)
sequences = tokenizer.texts_to_sequences(docs_list)
word_index = tokenizer.word_index # the dictionary
print(word_index)
print('Found %s unique tokens.' % len(word_index))
docs_words_index = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of samples:', docs_words_index.shape)
print('Sampele:(the zeros at the begining are for padding text to max length)')
# print(docs_words_index[2])


EMBEDDING_DIM = 100
print('Indexing word vectors.')
embeddings_index = {}
with open('glove.6B.100d.txt', encoding="utf8") as f:
    for line in f:
        values = line.split(sep=' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
# print('Found %s word vectors.' % len(embeddings_index))


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# print ('Shape of Embedding Matrix: ',embedding_matrix.shape)
# print(embedding_matrix)

docs_words_embeddings = np.zeros((len(docs_list), MAX_SEQUENCE_LENGTH, 100))

for i in range(len(docs_words_index)):
    for j in range(MAX_SEQUENCE_LENGTH):
        word_index = docs_words_index[i][j]
        docs_words_embeddings[i][j] = embedding_matrix[word_index]

docs_sentence_embeddings = np.zeros((len(docs_list), 100))

for i in range(len(docs_words_embeddings)):
    docs_sentence_embeddings[i] = np.sum(docs_words_embeddings[i], axis=0)

print(docs_sentence_embeddings[-1])

x_train, x_test, y_train, y_test = train_test_split(docs_sentence_embeddings, labels, test_size=0.20, random_state=1)

# # Train model
#
#
# logistic_regression = LogisticRegression()
# logistic_regression.fit(x_train, y_train)
#
# # Save model
#
# filename = 'sentence_embedding_sum.sav'
# pickle.dump(logistic_regression, open(filename, 'wb'))
#
#
# prediction = logistic_regression.predict(x_test)
# score = accuracy_score(y_test, prediction)
#
# print("accuracy: ")
# print(score)


# Load model
filename = 'sentence_embedding_sum.sav'

loaded_logistic_regression = pickle.load(open(filename, 'rb'))
prediction = loaded_logistic_regression.predict(x_test)
score = accuracy_score(y_test, prediction)

print("accuracy: ")
print(score)

print(loaded_logistic_regression.predict(docs_sentence_embeddings[:10]))
print(loaded_logistic_regression.predict(docs_sentence_embeddings[-10:]))


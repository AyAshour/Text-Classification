import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

review = input("review: ")

MAX_SEQUENCE_LENGTH = 1000

tokenizer = Tokenizer()
tokenizer.fit_on_texts([review])
sequences = tokenizer.texts_to_sequences([review])
word_index = tokenizer.word_index # the dictionary
print(word_index)
print('Found %s unique tokens.' % len(word_index))
docs_words_index = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

EMBEDDING_DIM = 100
print('Indexing word vectors.')
embeddings_index = {}
with open('glove.6B.100d.txt', encoding="utf8") as f:
    for line in f:
        values = line.split(sep=' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# print ('Shape of Embedding Matrix: ',embedding_matrix.shape)
# print(embedding_matrix)

review_words_embeddings = np.zeros((1, MAX_SEQUENCE_LENGTH, 100))


for j in range(MAX_SEQUENCE_LENGTH):
    word_index = docs_words_index[0][j]
    review_words_embeddings[0][j] = embedding_matrix[word_index]

review_sentence_embeddings = np.zeros((1, 100))

review_sentence_embeddings[0] = np.average(review_words_embeddings[0], axis=0)

filename = 'sentence_embedding_avg.sav'

loaded_logistic_regression = pickle.load(open(filename, 'rb'))
prediction = loaded_logistic_regression.predict(review_sentence_embeddings)
if prediction[0] == 1:
    print("positive")
else:
    print("negative")
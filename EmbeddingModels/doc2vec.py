from gensim import models
import nltk
import numpy as np
import os
import collections
from keras.preprocessing.text import text_to_word_sequence
from gensim.models.doc2vec import Doc2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from sklearn.utils import shuffle


positive_files_names = os.listdir("txt_sentoken/pos")
negative_files_names = os.listdir("txt_sentoken/neg")

docs_list = []

for file_name in positive_files_names:
    file = open("txt_sentoken/pos/" + str(file_name), "r")
    docs_list.append(file.read())

for file_name in negative_files_names:
    file = open("txt_sentoken/neg/" + str(file_name), "r")
    docs_list.append(file.read())

labels_positive = [1] * len(positive_files_names)
labels_negative = [0] * len(negative_files_names)

labels = labels_positive + labels_negative
labels = np.array(labels)

docs_tokens = []
for doc in docs_list:
    docs_tokens.append(text_to_word_sequence(doc))

Article = collections.namedtuple('Article', 'words tags paragraph')

tuples_list = []
for i in range(len(docs_tokens)):
    tuples_list.append(Article(words=docs_tokens[i], tags=[str(i)], paragraph=docs_list[i]))

tuples_list = shuffle(tuples_list)

# model = Doc2Vec(size=50,
#                 alpha=0.025,
#                 min_alpha=0.00025,
#                 min_count=1,
#                 dm =1)
#
# model.build_vocab(tuples_list)
#
# for epoch in range(20):
#     print('iteration {0}'.format(epoch))
#     model.train(tuples_list,
#                 total_examples=model.corpus_count,
#                 epochs=model.iter)
#     # decrease the learning rate
#     model.alpha -= 0.002
#     # fix the learning rate, no decay
#     model.min_alpha = model.alpha
#
# model.save("d2v.model")
# print("Model Saved")

model= Doc2Vec.load("d2v.model")

# Get sentences embeddings of train and test data

docs_sentence_embeddings = np.zeros((len(docs_list), 50))
for i in range(len(docs_sentence_embeddings)):
    docs_sentence_embeddings[i] = model.docvecs[str(i)]
print(docs_sentence_embeddings.shape)
#
x_train, x_test, y_train, y_test = train_test_split(docs_sentence_embeddings, labels, test_size=0.20, random_state=1)
#
# Train model

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)

# Save model

filename = 'Doc2Vec.sav'
pickle.dump(logistic_regression, open(filename, 'wb'))


prediction = logistic_regression.predict(x_test)
score = accuracy_score(y_test, prediction)

print("accuracy: ")
print(score)

print(logistic_regression.predict(docs_sentence_embeddings[:10]))
print(logistic_regression.predict(docs_sentence_embeddings[-10:]))

# Load model

filename = 'Doc2Vec.sav'

loaded_logistic_regression = pickle.load(open(filename, 'rb'))
prediction = loaded_logistic_regression.predict(x_test)
score = accuracy_score(y_test, prediction)

print("accuracy: ")
print(score)

print(loaded_logistic_regression.predict(docs_sentence_embeddings[:10]))
print(loaded_logistic_regression.predict(docs_sentence_embeddings[-10:]))

# putting things all together

# def doc2vec_predict(review):
#     tokens = text_to_word_sequence(review)
#     model = Doc2Vec.load("d2v.model")
#     doc2vec = model.infer_vector(tokens)
#     print(doc2vec)
#
#     filename = 'Doc2Vec.sav'
#     loaded_logistic_regression = pickle.load(open(filename, 'rb'))
#     y = loaded_logistic_regression.predict([doc2vec])
#     print(y)
#
#     if y[0] == 0:
#         print("this review is negative")
#     else:
#         print("this review is positive")
#
# x = input("Enter review: ")
# doc2vec_predict(x)
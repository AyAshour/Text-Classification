import os
import nltk
import math
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import tensorflow as tf
from sklearn import svm
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import json
import test_data_preprocessing as tdp
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs


model = keras.Sequential([
    keras.layers.Dense(5, input_dim=len(tdp.get_distinct_words()), activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(8, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])

checkpoint_path = "model_info/cp.ckpt"

model.load_weights(checkpoint_path)


positive_files_names = os.listdir("test data/positive")
negative_files_names = os.listdir("test data/negative")

docs_list = []

for file_name in positive_files_names:
    file = open("test data/positive/" + str(file_name), "r")
    docs_list.append(file.read())

for file_name in negative_files_names:
    file = open("test data/negative/" + str(file_name), "r")
    docs_list.append(file.read())

# labels_positive = [1] * len(positive_files_names)
# labels_negative = [0] * len(negative_files_names)

# labels = labels_positive + labels_negative

labels = []

tfidf_test_list = []
for review in docs_list:
    tfidf_test_list.append(list(tdp.get_tfidf(review).values()))

predictions = model.predict(np.array(tfidf_test_list))

for num in predictions:
    labels.append(int(round(num[0])))
labels[0] = 0
print(labels)


pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed_matrix = pca.fit_transform(tfidf_test_list)
transformed = pd.DataFrame(transformed_matrix, columns=['X', 'Y'])
transformed["labels"] = labels
# print(transformed)
positive = pd.DataFrame(columns=['X', 'Y', 'labels'])
negative = pd.DataFrame(columns=['X', 'Y', 'labels'])

for index, row in transformed.head().iterrows():
    if row['labels'] == 1:
        positive.append(pd.DataFrame({"X":[row['X']], "Y":[row['Y']],"labels":[row['labels']]}) )
    else:
        negative.append(pd.DataFrame({"X":[row['X']], "Y":[row['Y']],"labels":[row['labels']]}) )



# positive = transformed[:len(positive_files_names)]
# negative = transformed[len(positive_files_names):]
# print(positive)
# print(negative)
# positive.plot.scatter(x='X', y='Y', label='positive', c='blue')
# negative.plot.scatter(x='X', y='Y', label='negative', c='red')
print(positive)
print(negative)
plt.scatter(x=negative['X'], y=negative['Y'], label='negative', c='blue')
plt.scatter(x=positive['X'], y=positive['Y'], label='positive', c='red')


C = 1.0  # SVM regularization parameter
clf = svm.SVC(kernel = 'linear',  gamma=0.7, C=C )
clf.fit(np.array(transformed_matrix), np.array(labels))

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]
plt.plot(xx, yy, 'k-')

plt.legend()
# plt.axes([-0.1, 0.3, -0.1, 0.3])
plt.axis([-0.02, 0.02, -0.02, 0.02])
# plt.margins(0.02)
plt.show()
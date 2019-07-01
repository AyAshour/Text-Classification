import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import svm
from sklearn.decomposition import PCA as sklearnPCA


positive_files_names = os.listdir("train data/positive")
negative_files_names = os.listdir("train data/negative")

docs_list = []

for file_name in positive_files_names:
    file = open("train data/positive/" + str(file_name), "r")
    docs_list.append(file.read())

for file_name in negative_files_names:
    file = open("train data/negative/" + str(file_name), "r")
    docs_list.append(file.read())

vec = TfidfVectorizer()
vec.fit(docs_list)
x_train = vec.transform(docs_list)
# idf = vec.idf_
# print(len(dict(zip(vec.get_feature_names(), idf))))

# df = pd.DataFrame(x_train.toarray(), columns = vec.get_feature_names())
# tfidf_list = df.values

# tfidf_dicts_list = dict(zip(vec.get_feature_names(), idf))
# # print(tfidf_dict_list.values())

labels_positive = [1] * len(positive_files_names)
labels_negative = [0] * len(negative_files_names)

labels = labels_positive + labels_negative

x_train, labels = shuffle(x_train, labels)

nb = MultinomialNB()
nb.fit(x_train, labels)
score = nb.score(x_train, labels)
print("train: " , score)


positive_files_names = os.listdir("test data/positive")
negative_files_names = os.listdir("test data/negative")

docs_list = []

for file_name in positive_files_names:
    file = open("test data/positive/" + str(file_name), "r")
    docs_list.append(file.read())

for file_name in negative_files_names:
    file = open("test data/negative/" + str(file_name), "r")
    docs_list.append(file.read())

labels_positive = [1] * len(positive_files_names)
labels_negative = [0] * len(negative_files_names)

labels = labels_positive + labels_negative

x_test = vec.transform(docs_list)
df = pd.DataFrame(x_test.toarray(), columns = vec.get_feature_names())
tfidf_test_list = df.values

x_test, labels = shuffle(x_test, labels)

predictions = nb.predict(x_test)

print(predictions)
print("test acc: " ,metrics.accuracy_score(labels, predictions))

# ********************input*******************
x = input("enter review: ")
test = vec.transform([x])
y = nb.predict(test)
if y[0] == 0:
    print("this review is negative")
else:
    print("this review is positive")

# *******************plot******************
labels = []
for num in predictions:
    labels.append(int(round(num)))

# labels_positive = [1] * len(positive_files_names)
# labels_negative = [0] * len(negative_files_names)

# labels = labels_positive + labels_negative

pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed_matrix = pca.fit_transform(tfidf_test_list)
transformed = pd.DataFrame(transformed_matrix, columns=['X', 'Y'])
transformed["labels"] = labels
# print(transformed)

positive = pd.DataFrame(columns=['X', 'Y', 'labels'])
negative = pd.DataFrame(columns=['X', 'Y', 'labels'])

for index, row in transformed.iterrows():
    if row['labels'] == 1:
        positive = positive.append(pd.DataFrame({"X":[row['X']], "Y":[row['Y']],"labels":[row['labels']]}), ignore_index=True)
    else:
        negative = negative.append(pd.DataFrame({"X":[row['X']], "Y":[row['Y']],"labels":[row['labels']]}), ignore_index=True)


# positive = transformed[:len(positive_files_names)]
# negative = transformed[len(positive_files_names):]
# print(positive)
# print(negative)
# positive.plot.scatter(x='X', y='Y', label='positive', c='blue')
# negative.plot.scatter(x='X', y='Y', label='negative', c='red')

plt.scatter(x=negative['X'], y=negative['Y'], label='negative', c='red')
plt.scatter(x=positive['X'], y=positive['Y'], label='positive', c='blue')


C = 1.0  # SVM regularization parameter
clf = svm.SVC(kernel = 'linear',  gamma=0.7, C=C )
clf.fit(np.array(transformed_matrix), np.array(labels))

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]
plt.plot(xx, yy, 'k-')

plt.legend()
plt.axis([-0.02, 0.02, -0.02, 0.02])
plt.show()



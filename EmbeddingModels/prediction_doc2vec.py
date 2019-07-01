
from keras.preprocessing.text import text_to_word_sequence
from gensim.models.doc2vec import Doc2Vec
from sklearn.linear_model import LogisticRegression
import pickle

def doc2vec_predict(review):
    tokens = text_to_word_sequence(review)
    model = Doc2Vec.load("d2v.model")
    doc2vec = model.infer_vector(tokens)
    print(doc2vec)

    filename = 'Doc2Vec.sav'
    loaded_logistic_regression = pickle.load(open(filename, 'rb'))
    y = loaded_logistic_regression.predict([doc2vec])
    print(y)

    if y[0] == 0:
        print("this review is negative")
    else:
        print("this review is positive")

x = input("Enter review: ")
doc2vec_predict(x)
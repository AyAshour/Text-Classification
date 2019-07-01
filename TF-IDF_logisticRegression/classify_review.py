import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import nltk
import json
import math
import test_data_preprocessing as tdp

review = input("enter review: ")

x = list(tdp.get_tfidf(review).values())

model = keras.Sequential([
    keras.layers.Dense(10, input_dim=len(tdp.get_distinct_words()), activation=tf.nn.relu),
    keras.layers.Dense(8, activation=tf.nn.relu),
    keras.layers.Dense(8, activation=tf.nn.relu),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])

checkpoint_path = "model_info/cp.ckpt"

model.load_weights(checkpoint_path)
prediction = model.predict(np.array([x]))
print(prediction[0][0])


import json
import os

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn import (linear_model, metrics, model_selection, naive_bayes,
                     preprocessing, svm)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import settings
from keras.layers import *
from keras import layers, models, optimizers


def read_data(path_data):
    with open(path_data) as f:
        data = json.load(f)
    return data


labels = read_data(settings.DATA_LABEL)
contents = read_data(settings.DATA_CONTENT)

X_train, X_test, y_train, y_test = train_test_split(
    contents, labels, test_size=0.3, random_state=42)

# ===================== Feature Engineering =====================
# Tf-Idf Vectors as Features


# Word2vec
w2v = KeyedVectors.load_word2vec_format(
    settings.WORD2VEC_MODEL_PATH)
vocab = w2v.wv.vocab
wv = w2v.wv


def get_word2vec_data(X):
    word2vec_data = []
    for x in X:
        sentence = []
        for word in x.split(" "):
            if word in vocab:
                sentence.append(wv[word])

        word2vec_data.append(sentence)

    return word2vec_data


X_train_w2v = get_word2vec_data(X_train)
X_test_w2v = get_word2vec_data(X_test)


# ===================== Training =====================
# Label Encoder

encoder = preprocessing.LabelEncoder()

y_train_n = encoder.fit_transform(y_train)
y_test_n = encoder.fit_transform(y_test)

print(encoder.classes_)


def train_model(classifier, X_data, y_data, X_test, y_test, is_neuralnet=False, n_epochs=3):
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_data, y_data, test_size=0.1, random_state=42)

    if is_neuralnet:
        classifier.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=n_epochs, batch_size=512)

        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        val_predictions = val_predictions.argmax(axis=-1)
        test_predictions = test_predictions.argmax(axis=-1)
    else:
        classifier.fit(X_train, y_train)

        train_predictions = classifier.predict(X_train)
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)

    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))


def create_lstm_model():
    input_layer = Input(shape=(300,))

    layer = Reshape((10, 30))(input_layer)
    layer = LSTM(128, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)

    output_layer = Dense(10, activation='softmax')(layer)

    classifier = models.Model(input_layer, output_layer)

    classifier.compile(optimizer=optimizers.Adam(
    ), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return classifier

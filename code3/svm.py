# usr/bin/python3

import json
import re

import pandas as pd
import unicodedata2
from bs4 import BeautifulSoup
from pyvi import ViTokenizer
import gensim

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report

from gensim.models import KeyedVectors

import settings


def read_data(path_data):
    with open(path_data) as f:
        data = json.load(f)
    return data


# w2v = KeyedVectors.load_word2vec_format('data/vi/vi.vec')
# vocab = w2v.wv.vocab
# wv = w2v.wv


# def get_word2vec_data(X):
#     word2vec_data = []
#     for x in X:
#         sentence = []
#         for word in x.split(" "):
#             if word in vocab:
#                 sentence.append(wv[word])

#         word2vec_data.append(sentence)

#     return word2vec_data


# def train_model(classifier, X_data, y_data, X_test, y_test, is_neuralnet=False, n_epochs=3):
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_data, y_data, test_size=0.1, random_state=42)

#     if is_neuralnet:
#         classifier.fit(X_train, y_train, validation_data=(
#             X_val, y_val), epochs=n_epochs, batch_size=512)

#         val_predictions = classifier.predict(X_val)
#         test_predictions = classifier.predict(X_test)
#         val_predictions = val_predictions.argmax(axis=-1)
#         test_predictions = test_predictions.argmax(axis=-1)
#     else:
#         classifier.fit(X_train, y_train)

#         train_predictions = classifier.predict(X_train)
#         val_predictions = classifier.predict(X_val)
#         test_predictions = classifier.predict(X_test)

#     print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
#     print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))


# def create_lstm_model():
#     input_layer = Input(shape=(300,))

#     layer = Reshape((10, 30))(input_layer)
#     layer = LSTM(128, activation='relu')(layer)
#     layer = Dense(512, activation='relu')(layer)
#     layer = Dense(512, activation='relu')(layer)
#     layer = Dense(128, activation='relu')(layer)

#     output_layer = Dense(10, activation='softmax')(layer)

#     classifier = models.Model(input_layer, output_layer)

#     classifier.compile(optimizer=optimizers.Adam(
#     ), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     return classifier


if __name__ == "__main__":

    labels = read_data(settings.DATA_LABEL)
    contents = read_data(settings.DATA_CONTENT)

    X_train, X_test, y_train, y_test = train_test_split(
        contents, labels, test_size=0.3, random_state=42)

    # ==================================================================================

    my_tags = ['phương pháp', 'thái độ', 'cơ sở vật chất']

    # X_data_w2v = get_word2vec_data(X_train)
    # X_test_w2v = get_word2vec_data(X_test)

    # encoder = preprocessing.LabelEncoder()
    # y_train_n = encoder.fit_transform(y_train)
    # y_test_n = encoder.fit_transform(y_test)

    # ===================================================================================

    # PP1: Naive Bayes Classifier for Multinomial Models
    # nb = Pipeline([('vect', CountVectorizer()),
    #                ('tfidf', TfidfTransformer()),
    #                ('clf', MultinomialNB()),
    #                ])
    # nb.fit(X_train, y_train)

    # y_pred = nb.predict(X_test)

    # print('accuracy %s' % accuracy_score(y_pred, y_test))
    # print(classification_report(y_test, y_pred, target_names=my_tags))

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # PP2: Linear Support Vector Machine => 87
    # from sklearn.linear_model import SGDClassifier

    # sgd = Pipeline([('vect', CountVectorizer()),
    #                 ('tfidf', TfidfTransformer()),
    #                 ('clf', SGDClassifier(loss='hinge', penalty='l2',
    #                                       alpha=1e-3, random_state=42, max_iter=5, tol=None)),
    #                 ])
    # sgd.fit(X_train, y_train)

    # y_pred = sgd.predict(X_test)

    # print('accuracy %s' % accuracy_score(y_pred, y_test))
    # print(classification_report(y_test, y_pred, target_names=my_tags))

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # PP3: Logistic Regression => 84
    # from sklearn.linear_model import LogisticRegression

    # logreg = Pipeline([('vect', CountVectorizer()),
    #                    ('tfidf', TfidfTransformer()),
    #                    ('clf', LogisticRegression(n_jobs=1, C=1e5)),
    #                    ])
    # logreg.fit(X_train, y_train)

    # y_pred = logreg.predict(X_test)

    # print('accuracy %s' % accuracy_score(y_pred, y_test))
    # print(classification_report(y_test, y_pred, target_names=my_tags))

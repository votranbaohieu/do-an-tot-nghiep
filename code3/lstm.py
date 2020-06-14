import json
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from pyvi import ViTokenizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential


import settings

col_excel_content_name = 'nội dung ý kiến'
col_excel_label_name = 'class'


def clean_data_and_save(stopwords, acronyms):
    df = pd.concat(pd.read_excel(settings.DATA_TRAIN_PATH, sheet_name=[
                   0, 1, 2], usecols=[1, 2], nrows=400), ignore_index=True)

    df[col_excel_content_name] = df[col_excel_content_name].apply(
        lambda x: clean_text(x, stopwords, acronyms))

    df.drop_duplicates(keep=False, inplace=True)

    df[col_excel_content_name].replace("", np.nan, inplace=True)

    df.dropna(subset=[col_excel_content_name], inplace=True)

    # df[col_excel_content_name] = df[col_excel_content_name].apply(
    #     lambda x: x.split(' '))

    df[col_excel_label_name].to_json(
        settings.DATA_LABEL, force_ascii=False,  orient='records')

    df[col_excel_content_name].to_json(
        settings.DATA_CONTENT, force_ascii=False,  orient='records')


def read_stopwords(path_stopwords):
    with open(path_stopwords, 'r') as f:
        stopwords = set([w.strip().replace(' ', '_')
                         for w in f.readlines()])

    return stopwords


def read_acronyms(path_acronyms_json):
    with open(path_acronyms_json) as f:
        acronyms = json.load(f)
    return acronyms


def clean_text(text, stopwords, acronyms):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

    t = text.lower()

    t = BeautifulSoup(t, 'html.parser').get_text()

    t = " ".join([x.strip(settings.SPECIAL_CHARACTER) for x in t.split()])

    print(t)

    # Chuẩn hóa láy âm tiết
    t = re.sub(r'(\D)\1+', r'\1', t)

    t = " " + t + " "

    for key in acronyms:
        for value in acronyms[key]:
            v = ' ' + value + ' '
            if v in t:
                t = t.replace(v, ' ' + key + ' ')

    # # Tách từ
    t = ViTokenizer.tokenize(t)

    t = [word for word in t.split() if word not in stopwords]

    return " ".join(t)


def read_data(path_data):
    with open(path_data) as f:
        data = json.load(f)
    return data


def get_word2vec_data(X, vocab, wv):
    word2vec_data = []
    for x in X:
        sentence = []
        for word in x.split(" "):
            if word in vocab:
                sentence.append(wv[word])

        word2vec_data.append(sentence)

    return word2vec_data


def create_model_word2vec(doc, path_model, settings):
    sentences = []
    for text in doc:
        sentences.append(text.split(' '))

    model = Word2Vec(
        sentences=sentences, size=settings['size'], window=settings['window'], workers=settings['workers'], min_count=settings['min_count'], sg=settings['sg'], iter=settings['iter'])

    model.save(path_model)


# size: Dimensionality of the word vectors.
# window: Maximum distance between the current and predicted word within a sentence.
# min_count: Ignores all words with total frequency lower than this.
# workers: Use these many worker threads to train the model (=faster training with multicore machines).
# sg: Training algorithm: 1 for skip-gram; otherwise CBOW.
word2vec_settings = {
    'size': 200,
    'window': 5,
    'workers': 4,
    'min_count': 2,
    'sg': 1,
    'iter': 10
}

if __name__ == "__main__":
    MAX_SEQUENCE_LENGTH = 20
    MAX_NB_WORDS = 2000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2

    # Clean data
    stopwords = read_stopwords(settings.STOP_WORDS)
    acronyms = read_acronyms(settings.ACRONYMS)

    # clean_data_and_save(stopwords, acronyms)

    t = "nguyen,thanh long"
    t = " ".join([x.strip(settings.SPECIAL_CHARACTER) for x in t.split()])

    print(t)
    # # Load data
    # labels = read_data(settings.DATA_LABEL)
    # contents = read_data(settings.DATA_CONTENT)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     contents, labels, test_size=0.3, random_state=42)

    # # Label Encoder
    # encoder = preprocessing.LabelEncoder()
    # y_train_n = encoder.fit_transform(y_train)
    # y_test_n = encoder.fit_transform(y_test)

    # # Create word2vec model
    # create_model_word2vec(
    #     X_train, settings.WORD2VEC_MODEL_PATH, word2vec_settings)

    # # Load word2vec model
    # w2vmodel = Word2Vec.load(settings.WORD2VEC_MODEL_PATH)

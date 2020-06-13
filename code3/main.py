import json
import re

import gensim
import numpy as np
import pandas as pd
import unicodedata2
from bs4 import BeautifulSoup
from gensim.models import KeyedVectors
from keras.layers import Dense, Dropout, Embedding, Flatten, Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from pyvi import ViTokenizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import load_model

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
    t = text.lower()

    t = ' '.join(t.split())

    t = BeautifulSoup(t, 'html.parser').get_text()

    # Chuẩn hóa láy âm tiết
    t = re.sub(r'(\D)\1+', r'\1', t)

    for key in acronyms:
        for value in acronyms[key]:
            if value in t:
                t = t.replace(value, key)

    # Tách từ
    t = ViTokenizer.tokenize(t)

    t = [x.strip(settings.SPECIAL_CHARACTER) for x in t.split()]

    t = [word for word in t if word not in stopwords]

    return " ".join(t)

# ============================================================================================================

# Word2vec



def read_data(path_data):
    with open(path_data) as f:
        data = json.load(f)
    return data


def max_length(lines):
    return max([len(s.split()) for s in lines])


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


def define_model(length, vocab_size):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 100)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 100)(inputs2)
    conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # channel 3
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size, 100)(inputs3)
    conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    # plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model


stopwords = read_stopwords(settings.STOP_WORDS)
acronyms = read_acronyms(settings.ACRONYMS)
clean_data_and_save(stopwords, acronyms)


# Load data
labels = read_data(settings.DATA_LABEL)
contents = read_data(settings.DATA_CONTENT)

X_train, X_test, y_train, y_test = train_test_split(
    contents, labels, test_size=0.3, random_state=42)

# ===================== Training =====================
# Label Encoder

encoder = preprocessing.LabelEncoder()

y_train_n = encoder.fit_transform(y_train)
y_test_n = encoder.fit_transform(y_test)

# create tokenizer
tokenizer = create_tokenizer(X_train)
# calculate max document length
length = max_length(X_train)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, X_train, length)
testX = encode_text(tokenizer, X_test, length)
print(trainX.shape, testX.shape)

# define model
model = define_model(length, vocab_size)
# fit model
model.fit([trainX, trainX, trainX], np.array(
    y_train_n), epochs=10, batch_size=16)
# save the model
model.save(settings.MODEL_PATH)

# load the model
model = load_model(settings.MODEL_PATH)

# evaluate model on training dataset
loss, acc = model.evaluate([trainX, trainX, trainX],
                           np.array(y_train_n), verbose=0)
print('Train Accuracy: %f' % (acc*100))

# evaluate model on test dataset dataset
loss, acc = model.evaluate([testX, testX, testX],
                           np.array(y_test_n), verbose=0)
print('Test Accuracy: %f' % (acc*100))

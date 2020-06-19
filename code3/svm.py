import json
import re

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from pyvi import ViTokenizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

import settings


col_excel_content_name = 'nội dung ý kiến'
col_excel_label_name = 'class'


def clean_data_and_save(stopwords, acronyms):
    df = pd.concat(pd.read_excel(settings.DATA_TRAIN_PATH, sheet_name=[
                   0, 1, 2], usecols=[1, 2], nrows=450), ignore_index=True)

    df[col_excel_content_name] = df[col_excel_content_name].apply(
        lambda x: clean_text(x, stopwords, acronyms))

    df.drop_duplicates(keep=False, inplace=True)

    df[col_excel_content_name].replace("", np.nan, inplace=True)

    df.dropna(subset=[col_excel_content_name], inplace=True)

    # df[col_excel_content_name] = df[col_excel_content_name].apply(lambda x: x.split(' '))

    df[col_excel_label_name].to_json(settings.DATA_LABEL, orient='records')

    df[col_excel_content_name].to_json(settings.DATA_CONTENT, orient='records')


def read_stopwords(path_stopwords):
    with open(path_stopwords, 'r', encoding="utf-8") as f:
        stopwords = set([w.strip().replace(' ', '_')
                         for w in f.readlines()])

    f.close()
    return stopwords


def read_acronyms(path_acronyms_json):
    with open(path_acronyms_json, encoding="utf-8") as f:
        acronyms = json.load(f)
    return acronyms


def clean_text(text, stopwords, acronyms):
    t = text.lower()

    t = ' '.join(t.split())

    t = BeautifulSoup(t, 'html.parser').get_text()

    # Chuẩn hóa láy âm tiết
    t = re.sub(r'(\D)\1+', r'\1', t)

    t = " ".join([x.strip(settings.SPECIAL_CHARACTER) for x in t.split()])

    t = ' ' + t + ' '

    for key in acronyms:
        for value in acronyms[key]:
            v = ' ' + value + ' '
            if v in t:
                t = t.replace(v, ' ' + key + ' ')

    # Tách từ
    t = ViTokenizer.tokenize(t)

    t = [word for word in t.split(' ') if word not in stopwords]

    return " ".join(t)


def read_data(path_data):
    with open(path_data, encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    stopwords = read_stopwords(settings.STOP_WORDS_PATH)
    acronyms = read_acronyms(settings.ACRONYMS_PATH)
    clean_data_and_save(stopwords=stopwords, acronyms=acronyms)

    data_content = read_data(settings.DATA_CONTENT)
    data_label = read_data(settings.DATA_LABEL)

    Train_X, Test_X, Train_Y, Test_Y = train_test_split(data_content, data_label, test_size=0.3)

    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    Tfidf_vect = TfidfVectorizer(max_features=10000)
    Tfidf_vect.fit(data_content)

    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)


    # Naive Bayes
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X_Tfidf,Train_Y)
    predictions_NB = Naive.predict(Test_X_Tfidf)
    print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

    # SVM
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', random_state=0)
    SVM.fit(Train_X_Tfidf,Train_Y)
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y) * 100)
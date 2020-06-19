import json
import re
import os

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
import pickle

import settings

col_excel_content_name = 'nội dung ý kiến'
col_excel_label_name = 'class'

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
STOPWORDS_PATH= os.path.join(DIR_PATH, 'static/stopwords.txt') 
ACRONYMS_PATH= os.path.join(DIR_PATH, 'static/acronyms.json') 
SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''

def clean_data_and_save(path, stopwords, acronyms):
    df = pd.read_excel(path)

    df[col_excel_content_name] = df[col_excel_content_name].apply(
        lambda x: clean_text(x, stopwords, acronyms))

    df.drop_duplicates(keep=False, inplace=True)

    df[col_excel_content_name].replace("", np.nan, inplace=True)

    df.dropna(subset=[col_excel_content_name], inplace=True)

    labels = df[col_excel_label_name].tolist()
    contents = df[col_excel_content_name].tolist()

    return contents, labels 


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

    t = " ".join([x.strip(SPECIAL_CHARACTER) for x in t.split()])

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

def read_excel(path):
    df = pd.read_excel(path)
    df[col_excel_content_name].apply()
    print(df['class'].apply(lambda x: 1))

def main(algorithm, model_name):
    stopwords = read_stopwords(STOPWORDS_PATH)
    acronyms = read_acronyms(ACRONYMS_PATH)
    
    Train_X, Train_Y = clean_data_and_save(settings.TRAIN_PATH, stopwords, acronyms)
    Test_X, Test_Y = clean_data_and_save(settings.TEST_PATH, stopwords, acronyms)

    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    Tfidf_vect = TfidfVectorizer(max_features=2000, min_df=3, max_df=0.7)
    Tfidf_vect.fit(Train_X)

    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    model_path = 'src/app/models'

    if algorithm == 0:
        # Naive Bayes
        Naive = naive_bayes.MultinomialNB()
        Naive.fit(Train_X_Tfidf,Train_Y)
        
        # Save model
        pickle.dump(Naive, open(model_path + '/' + model_name.replace(' ', '_') + '.pkl', 'wb'))
        
        predictions_NB = Naive.predict(Test_X_Tfidf)
        return accuracy_score(predictions_NB, Test_Y) * 100

    elif algorithm == 1:
        # SVM
        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', random_state=0)
        SVM.fit(Train_X_Tfidf,Train_Y)

        # Save model
        pickle.dump(SVM, open(model_path + '/' + model_name.replace(' ', '_') + '.pkl', 'wb'))
        
        predictions_SVM = SVM.predict(Test_X_Tfidf)
        return accuracy_score(predictions_SVM, Test_Y) * 100

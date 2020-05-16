import re
import os
import pandas
import json
import numpy as np
import pandas as pd
from pyvi import ViTokenizer
from bs4 import BeautifulSoup
from gensim import corpora, matutils
import unicodedata2
import settings
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC


content_excel = 'nội dung ý kiến'
category_excel = 'class'


def read_excel_to_json(pathExcel, pathJson):
    df = pd.concat(pd.read_excel(pathExcel, skiprows=1, usecols=[1,2], sheet_name=[0,1,2]), ignore_index=True) 
    df.to_json(pathJson, orient='records')

def read_json(pathJson):
    with open(pathJson, encoding="utf-8") as f:
        s = json.load(f)
    return s

class FileReader(object):
    def __init__(self, filePath, encoder = None):
        self.filePath = filePath
        self.encoder = encoder if encoder != None else 'utf-16le'

    def read(self):
        with open(self.filePath) as f:
            s = f.read()
        return s

    def content(self):
        s = self.read()
        return s.decode(self.encoder)

    def read_json(self):
        with open(self.filePath) as f:
            s = json.load(f)
        return s

    def read_stopwords(self):
        with open(self.filePath, 'r') as f:
            stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
        return stopwords

    def load_dictionary(self):
        return corpora.Dictionary.load_from_text(self.filePath)

class FileStore(object):
    def __init__(self, filePath, data = None):
        self.filePath = filePath
        self.data = data

    def store_json(self):
        with open(self.filePath, 'w') as outfile:
            json.dump(self.data, outfile)

    def store_dictionary(self, dict_words):
        dictionary = corpora.Dictionary(dict_words)
        dictionary.filter_extremes(no_below=20, no_above=0.3)
        dictionary.save_as_text(self.filePath)

    # def save_pickle(self,  obj):
    #     outfile = open(self.filePath, 'wb')
    #     fastPickler = cPickle.Pickler(outfile, cPickle.HIGHEST_PROTOCOL)
    #     fastPickler.fast = 1
    #     fastPickler.dump(obj)
    #     outfile.close()

class NLP(object):
    def __init__(self, text=None):
        self.text = text
        self.__set_stopwords()

    def __set_stopwords(self):
        with open(settings.STOP_WORDS, 'r', encoding="utf8") as f:
            stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
        
        self.stopwords = stopwords

    def handle(self):
            t = self.text.lower()

            t = BeautifulSoup(t, 'html.parser').get_text()

            # Chuẩn hóa láy âm tiết
            t = re.sub(r'(\D)\1+', r'\1', t)

            # Tách từ
            t = ViTokenizer.tokenize(t)
            
            # Xóa dấu
            t = unicodedata2.normalize('NFD', t).encode('ascii', 'ignore').decode("utf-8")

            t = [x.strip(settings.SPECIAL_CHARACTER) for x in t.split()]

            t = [word for word in t if word not in self.stopwords]

            self.text = t

    def get_words_feature(self):
        self.handle()
        return self.text


class FeatureExtraction(object):
    def __init__(self, data):
        self.data = data

    def __build_dictionary(self):
        print('Building dictionary')
        dict_words = []
        i = 0
        for text in self.data:
            i += 1
            print("Step {} / {}".format(i, len(self.data)))
            words = NLP(text = text[content_excel]).get_words_feature()
            dict_words.append(words)
        FileStore(filePath=settings.DICTIONARY_PATH).store_dictionary(dict_words)

    def __load_dictionary(self):
        if os.path.exists(settings.DICTIONARY_PATH) == False:
            self.__build_dictionary()
        self.dictionary = FileReader(settings.DICTIONARY_PATH).load_dictionary()

    def __build_dataset(self):
        self.features = []
        self.labels = []
        i = 0
        for d in self.data:
            i += 1
            print("Step {} / {}".format(i, len(self.data)))
            self.features.append(self.get_dense(d[content_excel]))
            self.labels.append(d[category_excel])

    def get_dense(self, text):
        self.__load_dictionary()
        words = NLP(text).get_words_feature()
        # Bag of words
        vec = self.dictionary.doc2bow(words)
        dense = list(matutils.corpus2dense([vec], num_terms=len(self.dictionary)).T[0])
        return dense

    def get_data_and_label(self):
        self.__build_dataset()
        return self.features, self.labels


class Classifier(object):
    def __init__(self, features_train = None, labels_train = None, features_test = None, labels_test = None,  estimator = LinearSVC(random_state=0)):
        self.features_train = features_train
        self.features_test = features_test
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.estimator = estimator

    def training(self):
        self.estimator.fit(self.features_train, self.labels_train)
        self.__training_result()

    # def save_model(self, filePath):
    #     FileStore(filePath=filePath).save_pickle(obj=est)

    def __training_result(self):
        y_true, y_pred = self.labels_test, self.estimator.predict(self.features_test)
        print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    # read_excel_to_json(settings.DATA_TRAIN_PATH, settings.DATA_TRAIN_JSON)
    json_data = read_json(settings.DATA_TRAIN_JSON)

    json_train = []
    json_test = []

    for i, d in enumerate(json_data):
        if i % 3 and i != 0:
            json_train.append(d)
        else:
            json_test.append(d)

    
    features_train, labels_train = FeatureExtraction(data=json_train).get_data_and_label()
    features_test, labels_test = FeatureExtraction(data=json_test).get_data_and_label()

    # print(labels_test)

    est = Classifier(features_train=features_train, features_test=features_test, labels_train=labels_train, labels_test=labels_test)
    est.training()
    # est.save_model(filePath='trained_model/linear_svc_model.pk')
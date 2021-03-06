import re
import os
import pandas
import json
import numpy as np
import pandas as pd
from pyvi import ViTokenizer
from bs4 import BeautifulSoup
import unicodedata2
import settings

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_TRAIN_PATH = os.path.join(DIR_PATH, 'data/train/data2.xlsx')
DATA_TRAIN_JSON = os.path.join(DIR_PATH, 'data/train/data2.json')
STOP_WORDS_PATH = os.path.join(DIR_PATH, 'stopwords.txt')
SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''

def read_excel_to_json(pathExcel, pathJson):
    df = pd.concat(pd.read_excel(pathExcel, skiprows=1, usecols=[1,2], sheet_name=[0,1,2]), ignore_index=True) 
    df.to_json(pathJson, orient='records')

def read_json(pathJson):
    with open(pathJson, encoding="utf-8") as f:
        s = json.load(f)
    return s

class DocPreprocess(object):
    def __init__(self, data, label, content):
        self.data = data
        self.label = label
        self.content = content

        self.html = True
        self.stopwords = True
        self.accented_char = True
        self.special_char = True

        self.__set_stopwords()


    def __set_stopwords(self):
        with open(STOP_WORDS_PATH, 'r', encoding="utf8") as f:
            stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
        self.list_stopword = stopwords

    def remove_html(self):
        self.html = True
        return self
        
    def remove_stopwords(self):
        self.stopwords = True
        return self

    def remove_accented_char(self):
        self.accented_char = True
        return self

    def remove_special_char(self):
        self.special_char = True
        return self

    def handle(self):

        newData = []
        
        for v in self.data:
            t = v[self.content].lower()

            if self.html:
                t = BeautifulSoup(t, 'html.parser').get_text()

            # Chuẩn hóa láy âm tiết
            t = re.sub(r'(\D)\1+', r'\1', t)

            # Tách từ
            t = ViTokenizer.tokenize(t)

            if self.accented_char:
                t = unicodedata2.normalize('NFD', t).encode('ascii', 'ignore').decode("utf-8")

            if self.special_char:
                t = [x.strip(SPECIAL_CHARACTER) for x in t.split()]

            if self.stopwords:
                t = [word for word in t if word not in self.list_stopword]

            v[self.content] = t
            if v not in newData:
                newData.append(v)

        print(np.array(newData))


    def get_labels_and_contents(self):
        pass


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

    def save_pickle(self,  obj):
        outfile = open(self.filePath, 'wb')
        fastPickler = cPickle.Pickler(outfile, cPickle.HIGHEST_PROTOCOL)
        fastPickler.fast = 1
        fastPickler.dump(obj)
        outfile.close()

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

            t = [x.strip(SPECIAL_CHARACTER) for x in t.split()]

            t = [word for word in t if word not in self.stopwords]

            self.text = t

    def get_words_feature(self):
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
            words = NLP(text = text['content']).get_words_feature()
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
            self.features.append(self.get_dense(d['content']))
            self.labels.append(d['category'])

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

if __name__ == "__main__":
    read_excel_to_json(DATA_TRAIN_PATH, DATA_TRAIN_JSON)
    data_train = read_json(DATA_TRAIN_JSON)
    DocPreprocess(data_train, 'class', 'nội dung ý kiến').handle()
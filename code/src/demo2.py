import re
import os
import pandas as pd
import json
import numpy as np
from pyvi import ViTokenizer
from bs4 import BeautifulSoup
import unicodedata2

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_TRAIN_PATH = os.path.join(DIR_PATH, 'data/train/data.xlsx')
DATA_TRAIN_JSON = os.path.join(DIR_PATH, 'data/train/data.json')
STOP_WORDS_PATH = os.path.join(DIR_PATH, 'stopwords.txt')
SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''

label_name = 'category'
content_name = 'content'

# Read And Write File

def write_excel_to_json(pathExcel, pathJson):
    df = pd.read_excel(pathExcel)
    df.to_json(pathJson, orient="records")

def read_json(pathJson):
    with open(pathJson, encoding="utf-8") as f:
        s = json.load(f)
    return s

def read_stopwords(pathStopWords):
    with open(pathStopWords, 'r') as f:
        stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
    return stopwords


class DocPreprocess(object):
    def __init__(self, data, labelCol, contentCol):
        self.data = data
        self.labelCol = labelCol
        self.contentCol = contentCol
        
        self.html_stripping = False
        self.accented_char_removal = False
        self.special_char_removal = False
        self.stopword_removal = False

        self.__set_stopwords()

        # self.contraction_expansion = False
        # self.text_lemmatization = False

    def __set_stopwords(self):
        with open(STOP_WORDS_PATH, 'r') as f:
            stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
        self.stopwords = stopwords

    def html_stripping(self):
        self.html_stripping = True
        return self

    def remove_special_characters(self):
        self.special_char_removal = True
        return self

    def remove_accented_chars(self):
        self.accented_char_removal = True
        return self

    def remove_stopwords(self):
        self.stopword_removal = True
    
    def handle(self):        
        label = self.labelCol
        content = self.contentCol
        newData = []

        # print(self.data)

        for v in self.data:
            t = v[content].lower()
            
            if self.html_stripping:
                t = BeautifulSoup(t, 'html.parser').get_text()
            
            # Chuẩn hóa láy âm tiết
            t = re.sub(r'(\D)\1+', r'\1', t)

            # Tách từ
            t = ViTokenizer.tokenize(t)

            if self.remove_accented_chars:
                t = unicodedata2.normalize('NFD', t).encode('ascii', 'ignore').decode("utf-8")

            if self.remove_special_characters:
                t = [x.strip(SPECIAL_CHARACTER) for x in t.split()]
                t = [i for i in t if i]

            if self.remove_stopwords:
                stopwords = self.stopwords
                t = [word for word in t if word not in self.stopwords]
                

            v[content] = t
            if v not in newData:
                newData.append(v)
        
        print(np.array(newData))
        


    def get_labels_contents(self):
        self.handle()

        labels = []
        contents = []

        return labels, contents


def main():
    pd.read_excel(DATA_TRAIN_PATH).to_json(DATA_TRAIN_JSON, force_ascii=False, orient="records")
    training = read_json(DATA_TRAIN_JSON)
    # print(training[content_name].apply(lambda x: x.lower()))
    
    # labels, contents = DocPreprocess(training, label_name, content_name)
    DocPreprocess(training, label_name, content_name).handle()

if __name__ == "__main__":
    main()

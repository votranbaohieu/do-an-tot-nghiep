import os
import pandas as pd
import json
import numpy as np
from pyvi import ViTokenizer
from bs4 import BeautifulSoup

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_TRAIN_PATH = os.path.join(DIR_PATH, 'data/train/data.xlsx')
DATA_TRAIN_JSON = os.path.join(DIR_PATH, 'data/train/data.json')
STOP_WORDS = os.path.join(DIR_PATH, 'stopwords.txt')
SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''

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


# Preprocess

def strip_html_tags(text):
    return BeautifulSoup(text, 'html.parser').get_text()

def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text

def segmentation(text):
    return ViTokenizer.tokenize(text)

def remove_stopwords(text):
    tokens = segmentation(text)
    tokens = [x.strip(SPECIAL_CHARACTER).lower() for x in tokens.split()]

def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True)


class NLP(object):
    def __init__(self, text = None, stopwords = {}):
        self.text = text       
        self.stopwords = stopwords 

    def strip_html_tags(self):
        self.text = BeautifulSoup(self.text, 'html.parser').get_text()
        return self

    def segmentation(self):
        self.text = ViTokenizer.tokenize(self.text)
        return self

    def split_words(self):
        try:
            r = [x.strip(SPECIAL_CHARACTER).lower() for x in text.split()]
            return [i for i in r if i]
        except TypeError:
            return []

    def remove_stopwords(self, is_lower_case=True):
        pass

    def get_words_feature(self):
        split_words = self.split_words()
        return [word for word in split_words if word not in self.stopwords]

class DocPreprocess(object):    
    def __init__(self, data):
        self.data = data

    # def to_lower():
    #     for row in self.data:
    def total(self):
        newData = []
        for i, row in enumerate(self.data):
            self.data[i]['content'] = NLP(row['content']).split_words()
            if self.data[i] not in newData:
                newData.append(self.data[i])

        return newData        

    def get_data_and_label(self):
        pass


def main():
    # write_excel_to_json(DATA_TRAIN_PATH, DATA_TRAIN_JSON)
    data_train = read_json(DATA_TRAIN_JSON)
    stopwords = read_stopwords(STOP_WORDS)
    newData = DocPreprocess(data_train).total()

    print(np.array(newData))

if __name__ == "__main__":
    main()  

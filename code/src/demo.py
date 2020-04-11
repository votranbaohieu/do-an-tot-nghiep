import os
import pandas as pd
import json
import numpy as np
from pyvi import ViTokenizer

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_TRAIN_PATH = os.path.join(DIR_PATH, 'data/train/data.xlsx')
DATA_TRAIN_JSON = os.path.join(DIR_PATH, 'data/train/data.json')
SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''


def write_excel_to_json(pathExcel, pathJson):
    df = pd.read_excel(pathExcel)
    df.to_json(pathJson, orient="records")
    # return df.to_json(force_ascii=False, orient="records")


def read_json(pathJson):
    with open(pathJson, encoding="utf-8") as f:
        s = json.load(f)
    return s


class NLP(object):
    def __init__(self, text = None):
        self.text = text

    def __set_stopwords(self):
        self.stopwords = []

    def segmentation(self):
        return ViTokenizer.tokenize(self.text)

    def split_words(self):
        text = self.segmentation()
        try:
            return [x.strip(SPECIAL_CHARACTER).lower() for x in text.split()]
        except TypeError:
            return []


class DocPreprocess(object):
    def __init__(self, data, stop_words = None):
        self.data = data
        self.stop_words = stop_words



class FeatureExtraction(object):
    def __init__(self, data):
        self.data = data

    def get_data_and_label(self):
        pass


def main():
    temp = u"Nguyễn Thành Long"
    print(NLP(temp).split_words())

    # write_excel_to_json(DATA_TRAIN_PATH, DATA_TRAIN_JSON)
    # data_train = read_json(DATA_TRAIN_JSON)

    # print(np.array(data_train))


if __name__ == "__main__":
    main()

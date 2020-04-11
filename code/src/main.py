import os
import pandas as pd
import numpy as np
import json
from bs4 import BeautifulSoup
from pyvi import ViTokenizer

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_TRAIN_PATH = os.path.join(DIR_PATH, 'data/train/data.xlsx')
DATA_TRAIN_JSON = os.path.join(DIR_PATH, 'data/train/data.json')

# Doc file


class FileReader(object):
    def __init__(self, filePath, encoder=None):
        self.filePath = filePath
        self.encoder = encoder

    # def read_excel_to_json(self):
    #     df = pd.read_excel(self.filePath)
    #     return df.to_json()

    def read_json(self):
        with open(self.filePath, encoding=self.encoder) as f:
            s = json.load(f)
        return s

# Luu tru file


class FileStore(object):
    def __init__(self, filePath, data=None):
        self.filePath = filePath
        self.data = data

    def store_json(self):
        with open(self.filePath, 'w') as outfile:
            json.dump(self.data, outfile)


class TienXuLy(object):
    def __init__(self, data):
        self.data = data

    def remove_duplicate(self):
        newList = [self.data[0]]
        for e in self.data:
            if e not in newList:
                newList.append(e)
        self.data = newList
        return self

    def remove_tag_html(self):
        for i, v in enumerate(self.data):
            self.data[i]['content'] = BeautifulSoup(v['content'], 'html.parser').get_text()
        return self

    def Tokenizer(self):
        pass

    def toLowerCase(self):
        for i, v in enumerate(self.data):
            self.data[i]['content'] = v['content'].lower()
        return self

    def total(self):
        for i, v in enumerate(self.data):
            t = v['content']

            # Lowercase
            t = t.lower()

            # Remove html tag
            t = BeautifulSoup(t, 'html.parser').get_text()
            self.data[i]['content'] = t

        return self

    def toData(self):
        return self.data


def main():
    # Get data from excel to json
    # df = pd.read_excel(DATA_TRAIN_PATH)
    # df.to_json(DATA_TRAIN_JSON, orient="records")

    # Read data train
    train_loader = FileReader(filePath=DATA_TRAIN_JSON, encoder="utf-8")
    data_train = train_loader.read_json()

    # TIEN XU LY
    data_train = TienXuLy(data_train).total().remove_duplicate().toData()

    print(np.array(data_train))
    # print(len(data_train))


if __name__ == "__main__":
    main()

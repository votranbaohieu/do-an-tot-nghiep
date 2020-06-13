import json
import re

import pandas as pd
import unicodedata2
from bs4 import BeautifulSoup
from pyvi import ViTokenizer
import gensim

import settings

col_excel_content_name = 'nội dung ý kiến'
col_excel_label_name = 'class'


def clean_data_and_save(stopwords, acronyms):
    df = pd.concat(pd.read_excel(settings.DATA_TRAIN_PATH, sheet_name=[
                   0, 1, 2], usecols=[1, 2], nrows=400), ignore_index=True)

    df[col_excel_content_name] = df[col_excel_content_name].apply(
        lambda x: clean_text(x, stopwords, acronyms))

    df[col_excel_label_name].to_json(
        settings.DATA_LABEL, force_ascii=False,  orient='records')
    df[col_excel_content_name].to_json(
        settings.DATA_CONTENT, force_ascii=False,  orient='records')

    print(df[col_excel_content_name].apply(lambda x: len(x.split(' '))).sum())


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

    # Xóa dấu
    # Tách từ
    t = ViTokenizer.tokenize(t)

    # t = unicodedata2.normalize('NFD', t).encode(
    #     'ascii', 'ignore').decode("utf-8")

    t = [x.strip(settings.SPECIAL_CHARACTER) for x in t.split()]

    t = [word for word in t if word not in stopwords]

    return " ".join(t)


if __name__ == "__main__":
    stopwords = read_stopwords(settings.STOP_WORDS)
    acronyms = read_acronyms(settings.ACRONYMS)
    clean_data_and_save(stopwords, acronyms)

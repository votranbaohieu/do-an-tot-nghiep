import pandas as pd
import json
from pyvi import ViTokenizer
from gensim import corpora, matutils
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
import _pickle as cPickle
import re
import os
# cac duong dan

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_TRAIN_PATH = os.path.join(DIR_PATH, 'data/train/train.xlsx')
DATA_TRAIN_JSON = os.path.join(DIR_PATH, 'data/train/data_train.json')
DATA_TEST_PATH = os.path.join(DIR_PATH, 'data/train/test.xlsx')
DATA_TEST_JSON = os.path.join(DIR_PATH, 'data/train/data_test.json')
STOP_WORDS = os.path.join(DIR_PATH, 'stopwords.txt')
SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''
DICTIONARY_PATH = 'dictionary.txt'
filename = r'trained_model\\linear_svc_model4.pk'


# chuyen excel sang json
def write_excel_to_json(pathExcel, pathJson):
    df = pd.read_excel(pathExcel)
    df.to_json(pathJson, orient="records")


# doc json
def read_json(pathJson):
    with open(pathJson, encoding="utf-8") as f:
        s = json.load(f)
    return s


# doc tu dien stopword
def read_stopwords(pathStopWords):
    with open(pathStopWords, 'r', encoding="utf-8") as f:
        stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
    return stopwords

# tach tu tieng  viet and chuan hoa tu
def segmentation(text):
        text =ViTokenizer.tokenize(text)
        return re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)

def split_words(text):
        _text = segmentation(text)
        try:
            r = [x.strip(SPECIAL_CHARACTER).lower() for x in _text.split()]
            return [i for i in r if i]
        except TypeError:
            return []

def get_words_feature( text):
        split_word = split_words(text)
        return [word for word in split_word if word.encode('utf-8') not in read_stopwords(STOP_WORDS)]

def get_word(text):
        # tien xu ly
        word = get_words_feature(text)
        return word


def __build_dictionary(data):
        print ('Building dictionary')
        dict_words = []
        i = 0
        for text in data:
            i += 1
            #print("Step{} / {}".format(i, len(data)))
            words = get_word(text['content'])
            dict_words.append(words)
        dictionary = corpora.Dictionary(dict_words)
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        # luu file dictonary
        dictionary.save_as_text(DICTIONARY_PATH)
        return dictionary

def build_dataset(data):
        features = []
        labels = []
        # chay file data_train
        i = 0
        for d in data:
            i += 1
            #print("Step {} / {}".format(i, len(data)))
            features.append(get_dense(d['content']))
            labels.append(d['category'])
        return  features,labels

def get_dense(text):
    dictionary= corpora.Dictionary.load_from_text(DICTIONARY_PATH)
    words = get_word(text)
    #print(words)
    # Bag of words
    vec = dictionary.doc2bow(words)
    #print(len(vec))
    #print(vec)
    dense = list(matutils.corpus2dense([vec], num_terms=len(dictionary)).T[0])
    #print(len(dense))
    return dense

def get_data_and_label(data):
    features, labels = build_dataset(data)
    return features, labels

def Classifier(features_train = None, features_test = None,labels_train = None, labels_test = None,  estimator = LinearSVC(random_state=0, dual= False, C= 1.0)): #ramdom_state=0
        features_train = features_train
        features_test = features_test
        labels_train = labels_train
        labels_test = labels_test
        estimator = estimator
        return features_train, features_test, labels_train, labels_test, estimator

def training(estimator, features_train, labels_train, features_test, labels_test):
        estimator.fit(features_train, labels_train)
        __training_result(features_test, labels_test, estimator)

def save_model(filePath, obj, data =None):
    with open(filePath, 'w') as outfile:
        json.dump(data, outfile)
    outfile = open(filePath, 'wb')
    fastPickler = cPickle.Pickler(outfile, -1)
    fastPickler.fast = 1
    fastPickler.dump(obj)
    outfile.close()

# đánh giá độ chính xác
def __training_result(features_test,labels_test,estimator ):
        y_true, y_pred = labels_test, estimator.predict(features_test)
        print(classification_report(y_true, y_pred))

def main():
    write_excel_to_json(DATA_TRAIN_PATH, DATA_TRAIN_JSON)
    data_train = read_json(DATA_TRAIN_JSON)
    write_excel_to_json(DATA_TEST_PATH, DATA_TEST_JSON)
    data_test = read_json(DATA_TEST_JSON)
    __build_dictionary(data_train)
    features_train, labels_train =get_data_and_label(data_train)
    features_test, labels_test = get_data_and_label(data_test)
    features_train, features_test,labels_train, labels_test, estimator = Classifier(features_train, features_test, labels_train, labels_test)
    training(estimator, features_train, labels_train, features_test, labels_test)
    save_model(filename,obj=estimator)
    
if __name__ == "__main__":
    main()


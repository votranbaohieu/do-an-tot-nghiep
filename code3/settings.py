import os

# DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# DATA_TRAIN_PATH = os.path.join(DIR_PATH, 'data/10_cate/train/')
# DATA_TEST_PATH = os.path.join(DIR_PATH, 'data/10_cate/test/')
# DATA_TRAIN_JSON = os.path.join(DIR_PATH, 'data_train.json')
# DATA_TEST_JSON = os.path.join(DIR_PATH, 'data_test.json')
# STOP_WORDS = os.path.join(DIR_PATH, 'stopwords-nlp-vi.txt')
# SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''
# DICTIONARY_PATH = 'dictionary.txt'


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_TRAIN_PATH = os.path.join(DIR_PATH, 'data/data.xlsx')
DATA_TRAIN_JSON = os.path.join(DIR_PATH, 'data/data.json')
SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|\n\t\''

DATA_LABEL = os.path.join(DIR_PATH, 'data/data_label.json')
DATA_CONTENT = os.path.join(DIR_PATH, 'data/data_content.json')

STOP_WORDS = os.path.join(DIR_PATH, 'data/stopwords.txt')
ACRONYMS = os.path.join(DIR_PATH, 'data/acronyms.json')
# DICTIONARY_PATH = os.path.join(DIR_PATH, 'dictionary.txt')

WORD2VEC_MODEL_PATH = os.path.join(DIR_PATH, 'models/word2vec.model')
MODEL_PATH = os.path.join(DIR_PATH, 'models/model.h5')

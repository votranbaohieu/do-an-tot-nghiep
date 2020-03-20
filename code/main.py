from pyvi import ViTokenizer


class NLP(object):
    def __init__(self, text=None):
        self.text = text
        # self.__set_stopwords()

    def segmentation(self):
        return ViTokenizer.tokenize(self.text)


temp = u"Chào các bạn tôi là Phạm Văn Toàn đến từ blog Tự học Machine Learning"

print(NLP(text=temp).segmentation())

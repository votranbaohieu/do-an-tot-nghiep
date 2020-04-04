import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_TRAIN_PATH = os.path.join(DIR_PATH, 'data/train/data.xlsx')
DATA_TRAIN_JSON = os.path.join(DIR_PATH, 'data/train/data.json')


class TienXuLy(object):
    def __init__(self, data):
        self.data = data


    def remove_duplicate_json(self):
        # newList = [self.data[0]]
        pass



def main():
    pass


if __name__ == "__main__":
    main()
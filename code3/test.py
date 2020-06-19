import pandas as pd

import settings

pp = pd.read_excel(settings.DATA_TRAIN_PATH, sheet_name=[0], usecols=[1, 2])
td = pd.read_excel(settings.DATA_TRAIN_PATH, sheet_name=[1], usecols=[1, 2])
csvc = pd.read_excel(settings.DATA_TRAIN_PATH, sheet_name=[2], usecols=[1, 2])

trainX = []
testX = []

trainContent = []
testContent = []

trainLabel = []
testLabel = []

col_excel_content_name = 'nội dung ý kiến'
col_excel_label_name = 'class'

# print(type(pp[0]['nội dung ý kiến'][3::3].array))
t = 1
for index, row in enumerate(pp[0]['nội dung ý kiến']):    
    if t == 3:
        t = 1
        testContent.append(row)
        testLabel.append('phương pháp')
    else:
        t+=1
        trainContent.append(row)
        trainLabel.append('phương pháp')

t = 1
for index, row in enumerate(td[1]['nội dung ý kiến']):    
    if t == 3:
        t = 1
        testContent.append(row)
        testLabel.append('thái độ')
    else:
        t+=1
        trainContent.append(row)
        trainLabel.append('thái độ')

t = 1
for index, row in enumerate(csvc[2]['nội dung ý kiến']):    
    if t == 3:
        t = 1
        testContent.append(row)
        testLabel.append('cơ sở vật chất')
    else:
        t+=1
        trainContent.append(row)
        trainLabel.append('cơ sở vật chất')

trainX = {
    col_excel_label_name: trainLabel,
    col_excel_content_name: trainContent,
}

testX = {
    col_excel_label_name: testLabel,
    col_excel_content_name: testContent,
}

dfTrain = pd.DataFrame(trainX, columns=[col_excel_label_name, col_excel_content_name])
dfTest = pd.DataFrame(testX, columns=[col_excel_label_name, col_excel_content_name])

dfTrain.to_excel('data_train.xlsx')
dfTrain.to_excel('test_train.xlsx')
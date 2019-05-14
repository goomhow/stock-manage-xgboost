import numpy as np
import pandas as pd


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    r = set()
    for row in dataSet:
        r = r|set(row)
    return list(r)


def setOfWordsToVec(vocabList, input):
    return [1 if i in input else 0 for i in vocabList]

def trainNB0(train,label):
    data = pd.DataFrame(data=train)
    data['label'] = label
    data1 = data[data['label'] == 1]
    data0 = data[data['label'] == 0]
    del data1['label']
    del data0['label']
    d1 = data1.sum(axis=0)/float(sum(data1))
    d0 = data0.sum(axis=0) / float(sum(data0))
    rate = data1.shape[0]/float(data.shape[0])
    return d0.values, d1.values, rate

if __name__=='__main__':
    train, label = loadDataSet()
    vocabList = createVocabList(train)
    train = [setOfWordsToVec(vocabList,row) for row in train]
    print(trainNB0(train, label))

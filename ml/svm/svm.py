from sklearn import svm
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer, MinMaxScaler
l = []


def format_series(s, flag):
    if flag:
        max_ = max(s)
        min_ = min(s)
        l.append([min_, max_])
    else:
        arg = l.pop(0)
        min_ = arg[0]
        max_ = arg[1]

    def cal(i):
        return (i-min_)/(max_-min_)
    return [cal(i) for i in s]


if __name__ == '__main__':
    impt = Imputer()
    scal = MinMaxScaler()
    train = pd.read_csv("broadband_train.csv", sep=",", encoding='gbk')
    train['GENDER'] = train['GENDER'].replace('男', 0).replace('女', 1)
    train['AUTOPAY'] = train['AUTOPAY'].replace('否', 0).replace('是', 1)
    train = train.apply(lambda s: format_series(s, True))
    test = pd.read_csv("broadband_test.csv", sep=",", encoding='utf-8')
    test['GENDER'] = test['GENDER'].replace('男', 0).replace('女', 1)
    test['AUTOPAY'] = test['AUTOPAY'].replace('否', 0).replace('是', 1)
    test = test.apply(lambda s: format_series(s, False))
    train_X = train.iloc[:, 1:-1]
    train_Y = train.iloc[:, -1]
    test_X = test.iloc[:, 1:]

    impt.fit(train_X)
    train_X = impt.transform(train_X)
    test_X = impt.transform(test_X)

    scal.fit(train_X)
    train_X = scal.transform(train_X)
    test_X = scal.transform(test_X)

    model = svm.SVC()
    model.fit(train_X, train_Y)
    print(cross_val_score(model, train_X, train_Y))
    data = model.predict(test_X)
    print(data)
    #[ 0.83532934  0.84984985  0.81927711]
    #[ 0.83532934  0.84984985  0.81927711]
    #[ 0.83532934  0.84984985  0.81927711]



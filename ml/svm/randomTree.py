from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

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
    train = pd.read_csv("broadband_train.csv", sep=",", encoding='gbk')
    train['GENDER'] = train['GENDER'].replace('男', 0).replace('女', 1)
    train['AUTOPAY'] = train['AUTOPAY'].replace('否', 0).replace('是', 1)
    train = train.apply(lambda s: format_series(s, True))
    train.fillna(train.mean(),inplace=True)
    test = pd.read_csv("broadband_test.csv", sep=",", encoding='utf-8')
    test['GENDER'] = test['GENDER'].replace('男', 0).replace('女', 1)
    test['AUTOPAY'] = test['AUTOPAY'].replace('否', 0).replace('是', 1)
    test = test.apply(lambda s: format_series(s, False))
    test.fillna(test.mean(), inplace=True)
    train_X = train.iloc[:, 1:-1]
    train_Y = train.iloc[:, -1]
    test_X = test.iloc[:, 1:]
    print(train_X.shape[1])
    model = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    model.fit(train_X, train_Y)
    print(cross_val_score(model, train_X, train_Y))
    data = model.predict(test_X)
    print(data)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale,Imputer
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


def fill_mean(df):
    cols = df.columns
    for col in cols:
        if col in ['GENDER', 'AUTOPAY']:
            last = 0
            s = df[col]
            for i in s.index:
                if s[i]:
                    last = s[i]
                else:
                    s[i] = last
            df[col] = s
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df


if __name__ == '__main__':
    train = pd.read_csv("broadband_train.csv", sep=",", encoding='gbk')
    train['GENDER'] = train['GENDER'].replace('男', 0).replace('女', 1)
    train['AUTOPAY'] = train['AUTOPAY'].replace('否', 0).replace('是', 1)
    train = train.apply(lambda s: format_series(s, True))
    test = pd.read_csv("broadband_test.csv", sep=",", encoding='utf-8')
    test['GENDER'] = test['GENDER'].replace('男', 0).replace('女', 1)
    test['AUTOPAY'] = test['AUTOPAY'].replace('否', 0).replace('是', 1)
    test = test.apply(lambda s: format_series(s, False))
    train_X = fill_mean(train.iloc[:, 1:-1])
    train_Y = train.iloc[:, -1]
    test_X = fill_mean(test.iloc[:, 1:])
    im = Imputer(strategy='most_frequent')
    im.fit(train_X)
    train_X = im.transform(train_X)
    test_X = im.transform(test_X)
    model = DecisionTreeClassifier()
    model.fit(train_X, train_Y)
    print(cross_val_score(model, train_X, train_Y))
    data = model.predict(test_X)
    print(data)



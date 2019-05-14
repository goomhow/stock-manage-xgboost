import pandas as pd
import numpy as np


def shannon(s):
    s = s.value_counts()
    return (s / s.sum()).apply(lambda x: -1 * x * np.log2(x)).sum()


class DecisionTree(object):
    def __init__(self, data, sep='\t', names=None):
        self.sep=sep
        self.names=names
        self.data = pd.read_csv(data, sep=sep, names=names)
        self.len = self.data.shape[0]
        self.wide = self.data.shape[1]
        self.tree = {}

    def best_group(self, data):
        length, wide = data.shape
        base = shannon(self.data.iloc[:, -1])
        gain = 0.0
        aixs = -1

        def cal(df):
            s = df.iloc[:, -1].value_counts()
            return (s / s.sum()).apply(lambda x: -1 * x * np.log2(x)).sum()*df.shape[0]/length
        for i in range(wide-1):
            ci = data.groupby(by=self.data.iloc[:, i]).apply(cal).sum()
            new_gain = base-ci
            if base > gain:
                gain = new_gain
                aixs = i
        return aixs

    def __create_tree(self, data, root):
        aixs = self.best_group(data)
        aixs_name = self.names[aixs]
        root[aixs_name] = dict()
        for name, df in data.groupby(by=data.iloc[:, aixs]):
            df = df.drop(columns=aixs_name)
            T = set(df.iloc[:, -1])
            if len(T) == 1:
                root[aixs_name][name] = T.pop()
            else:
                self.__create_tree(df, root[aixs_name])

    def init_tree(self):
        self.__create_tree(self.data, self.tree)


if __name__ == '__main__':
    d = DecisionTree('lenses.txt', sep='\t', names=list('ABCDT'))
    print(d.best_group(d.data.iloc[:, [0, 1, 2, 4]]))
    d.init_tree()
    print(d.tree)
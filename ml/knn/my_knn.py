import numpy as np
import pandas as pd
import math
import gevent.monkey as m
m.patch_all()
import gevent.pool as pool
import multiprocessing


class KNN(object):
    def __init__(self, csv, sep='\t', names=None, k=7):
        self.k = k
        self.data = pd.read_csv(csv,sep=sep, names=names)
        self.len = self.data.shape[1]
        self.train_X = self.data.iloc[:, :-1]
        self.train_Y = self.data.iloc[:, -1]
        self.max = self.train_X.max().values
        self.min = self.train_X.min().values
        self.train_X = self.to_one(self.train_X)

    def to_one(self, X):
        return (X-self.min)/(self.max-self.min)

    def fit(self, X):
        X = self.to_one(X)
        return [self._one_to_many(list(i)) for i in X]

    def _one_to_many(self, x):
        a = pow(self.train_X - x, 2)
        a = a.sum(axis=1)
        a = a.sort_values()[:self.k].index
        t = self.train_Y[a]
        t = t.value_counts().index[0]
        return t

    def accurate_rate(self, sample_rate=0.1, times=100):
        n = math.floor(self.len*sample_rate)

        def process(t):
            data = self.data.sample(n)
            X = data.iloc[:, :-1].values
            Y = data.iloc[:, -1].values
            P = self.fit(X)
            cnt = 0
            tol = len(P)
            for i in range(tol):
                if P[i] == Y[i]:
                    cnt += 1
            return cnt / tol
        gPool = pool.Pool(times)
        pl = gPool.map(process, range(times))
        return np.mean(pl)

    def find_k(self, sample_rate=0.1, times=100):
        def find(i):
            self.k = i
            return (self.accurate_rate(sample_rate=sample_rate,times=times),i)
        pPool = multiprocessing.Pool()
        pl = pPool.map(find, range(1, 100))
        pl = sorted(pl, reverse=True)
        return pl[0][1]


if __name__ == '__main__':
    knn = KNN('datingTestSet.txt', names=list('ABCT'))
    print(knn.find_k())

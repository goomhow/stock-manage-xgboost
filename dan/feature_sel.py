from sklearn.feature_selection import *
from sklearn.externals import joblib
import pandas as pd
import time


def train_select_feature_model(model, X, y):
    if isinstance(model, str):
        model = joblib.load(model)
    rfe = RFECV(estimator=model, step=1, cv=5, scoring='f1', n_jobs=-1)
    rfe.fit(X, y)
    joblib.dump(rfe, rfe_name)


def select_feature(X):
    rfe = joblib.load(rfe_name)
    return rfe.transform(X)


def select_feature(model, X, y):
    SelectKBest(k=80)


if __name__ == '__main__':
    start = time.time()
    rfe_name = 'dan_cdma_rfe.pkl'
    data = pd.read_csv('dan_train_201806.csv', index_col='PRD_INST_ID')
    train_select_feature_model('CDanCdmaModel_101606.pkl', data.drop(columns='LABEL'), data['LABEL'])
    end = time.time()
    cost = int(end-start)/60
    print('cost:%dmin' % cost)

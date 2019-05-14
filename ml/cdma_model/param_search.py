from datetime import datetime
import numpy as np
import pandas as pd
from sklearn import metrics, learning_curve, svm
from sklearn.model_selection import *
from sklearn.externals import joblib
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import os


def get_transformed_data(fname='cdma_train.csv',frac=0.3):
    xName = fname.split(".")[0]+'_x.csv'
    if os.path.exists(xName):
        data = pd.read_csv(xName)
    else:
        data = pd.read_csv(fname)
        data = data[data.LABEL > -1]
        common = [i[:-2] for i in data.columns if i.endswith('_A') and not i.startswith('STD_PRD_INST_STAT_ID')]
        A = data[[i+'_A' for i in common]].rename(columns=lambda x: x[-2])
        B = data[[i+'_B' for i in common]].rename(columns=lambda x: x[-2])
        C = data[[i+'_C' for i in common]].rename(columns=lambda x: x[-2])
        B_A = B - A
        C_B = C - B
        data[[i+'_A' for i in common]] = B_A.rename(columns=lambda x: x+'_A')
        data[[i+'_B' for i in common]] = C_B.rename(columns=lambda x: x+'_B')
        data.to_csv(xName, index=False, header=True)
    d_train = data.sample(frac=frac)
    X = d_train.drop('LABEL', axis=1).drop("PRD_INST_ID", axis=1)
    y = d_train['LABEL']
    return data, X, y

data = pd.read_csv('cdma_train.csv')
data = data[data.LABEL > -1]
d_train = data.sample(frac=0.3)
X = d_train.drop('LABEL', axis=1).drop("PRD_INST_ID", axis=1)
y = d_train['LABEL']
xgtrain = xgb.DMatrix(X, label=y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
# 第一步 设置XGB params的初始值
params = {'learning_rate': 0.1,
 'n_estimators': 146,
 'max_depth': 14,
 'min_child_weight': 2,
 'gamma': 0,
 'subsample': 0.95,
 'colsample_bytree': 0.65,
 'scale_pos_weight': 7,
 'n_jobs': 32,
 'objective': 'binary:logistic',
 'reg_alpha': 0,
 'reg_lambda': 0.005}


def set_best_tree_size():
    cvresult = xgb.cv(params, xgtrain, num_boost_round=params['n_estimators'], nfold=5, metrics='auc',
                      early_stopping_rounds=50)
    params['n_estimators'] = cvresult.shape[0]
    print('Tree Size is {}'.format(params['n_estimators']))


def grid_search_param(param_grid):
    gsearch2 = GridSearchCV(estimator=XGBClassifier(**params), param_grid=param_grid, scoring='f1_weighted', cv=5)
    gsearch2.fit(X, y)
    print(gsearch2.best_params_, gsearch2.best_score_)


def modelfit(useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    alg = XGBClassifier(**params)
    df = data.sample(frac=0.3)
    pX = df.drop('LABEL', axis=1).drop("PRD_INST_ID", axis=1)
    py = df['LABEL']
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb_param['n_estimators'], nfold=cv_folds,
                          metrics='f1_weighted', early_stopping_rounds=early_stopping_rounds)
        print(cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    # Fit the algorithm on the data
    alg.fit(X, y, eval_metric='auc')
    y_pred = alg.predict(pX)
    accuracy = metrics.accuracy_score(py, y_pred)
    print("精确率Accuracy: %.2f%%" % (accuracy * 100.0))
    print('auc:', metrics.roc_auc_score(py, y_pred))
    train_report = metrics.classification_report(py, y_pred)
    print(train_report)
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    print(feat_imp)
    return alg

# 寻找最优 max_depth 和 min_child_weight
param_grid = {
     'max_depth':[5,6,7],
     'min_child_weight':[3,4,5]
}
grid_search_param(param_grid)

# 寻找最优 gamma
param_grid = {'gamma':[i/10.0 for i in range(0,5)]}
grid_search_param(param_grid)

#寻找最优 subsample 和 colsample_bytree
param_grid = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
grid_search_param(param_grid)

model = XGBClassifier(**params)
model.fit(X, y, eval_metric=metrics.f1_score)


def evaluate_model(model):
    edata=data.sample(frac=0.3)
    eX = edata.drop('LABEL', axis=1).drop("PRD_INST_ID", axis=1)
    ey = edata['LABEL']
    y_pred = model.predict(eX)
    accuracy = metrics.accuracy_score(ey, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('auc:', metrics.roc_auc_score(ey, y_pred))
    train_report = metrics.classification_report(ey, y_pred)
    print(train_report)
    feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
    print(feat_imp)


xgb.plot_tree()

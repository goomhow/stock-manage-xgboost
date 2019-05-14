import pandas as pd
from sklearn import metrics
from sklearn.model_selection import *
from sklearn.externals import joblib
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import os
import time

ID_COLUMN = 'PRD_INST_ID'
LABEL = 'LABEL'

def running_time(func):
    def wrapper(*params,**kwargs):
        startTime = time.time()
        x = func(*params,**kwargs)
        endTime = time.time()
        msecs = int(endTime - startTime)/60
        print("time is %d min" %msecs)
        return x
    return wrapper

@running_time
def get_transformed_data(month, frac=0.3):
    dpath = 'dan_train_%s.csv' % month
    if not os.path.exists(dpath):
        fname = 'data/train_%s.csv' % month
        lname = 'label/label_%s.csv' % month
        data = pd.read_csv(fname, index_col=ID_COLUMN)
        data.drop(columns='LABEL', inplace=True)
        label = pd.read_csv(lname, index_col=ID_COLUMN)
        df = label.join(data).dropna()
        del data
        del label
        df.to_csv(dpath, index=True, index_label=ID_COLUMN)
    else:
        df = pd.read_csv(dpath,index_col=ID_COLUMN)
    n = df.sample(frac=frac)
    X = n.drop(columns='LABEL')
    y = n['LABEL']
    return df, X, y

data,X,y = get_transformed_data('201805',0.1)
xgtrain = xgb.DMatrix(X, label=y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
# 第一步 设置XGB params的初始值
params = {'learning_rate': 0.1,
 'n_estimators': 2000,
 'max_depth': 9,
 'min_child_weight': 1,
 'gamma': 0,
 'subsample': 0.8,
 'colsample_bytree': 0.9,
 'scale_pos_weight': 10,
 'n_jobs': 50,
 'objective': 'binary:logistic',
 'reg_alpha': 0.005,
 'reg_lambda': 0.005}

@running_time
def set_best_tree_size():
    cvresult = xgb.cv(params, xgtrain, num_boost_round=params['n_estimators'], nfold=5, metrics='auc',
                      early_stopping_rounds=50)
    params['n_estimators'] = cvresult.shape[0]
    print('Tree Size is {}'.format(params['n_estimators']))

@running_time
def grid_search_param(param_grid):
    gsearch2 = GridSearchCV(estimator=XGBClassifier(**params), param_grid=param_grid, scoring='f1', cv=5)
    gsearch2.fit(X, y)
    print(gsearch2.best_params_, gsearch2.best_score_)
    return gsearch2.best_params_


@running_time
def modelfit(useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    alg = XGBClassifier(**params)
    df = data.sample(frac=0.3)
    pX = df.drop('LABEL', axis=1).drop("PRD_INST_ID", axis=1)
    py = df['LABEL']
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb_param['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
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


def update_best_params(params_grid):
    a = grid_search_param(params_grid)
    for k, v in params_grid.items():
        i = v.index(a[k])
        if i == 0:
            a[k] = [v[0], (v[1]+v[0])/2.0]
        elif i == len(v)-1:
            a[k] = [(v[-1]+v[-2])/2.0, v[-1]]
        else:
            a[k] = [(v[i - 1] + v[i]) / 2.0, v[i], (v[i] + v[i + 1]) / 2.0]
    params.update(grid_search_param(a))

# 寻找最优 max_depth 和 min_child_weight
param_grid = {
     'max_depth': range(3, 11),
     'min_child_weight': range(0, 9)
}
params.update(grid_search_param(param_grid))

# 寻找最优 gamma
param_grid = {'gamma': [i/10.0 for i in range(0, 5)]}
params.update(grid_search_param(param_grid))

#寻找最优 scale_pos_weight
param_grid = {'scale_pos_weight': [0,1,10,20,50,100]}
update_best_params(param_grid)

#寻找最优 subsample 和 colsample_bytree
param_grid = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
update_best_params(param_grid)
#寻找最优 reg_alpha
param_grid = {'reg_alpha': [0, 0.005, 0.01, 0.1, 1]}
update_best_params(param_grid)

#寻找最优 reg_lambda
param_grid = {'reg_lambda': [0, 0.005, 0.01, 0.1, 1]}
update_best_params(param_grid)

model = XGBClassifier(**params)
model.fit(X, y, eval_metric=metrics.f1_score)
feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)


def evaluate_model(model_params,X,y):
    model = XGBClassifier(**model_params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
    model.fit(X_train, y_train, eval_metric=metrics.f1_score)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print('predict size:%d' % y_pred[y_pred == 1].shape[0])
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('auc:', metrics.roc_auc_score(y_test, y_pred))
    train_report = metrics.classification_report(y_test, y_pred)
    print(train_report)
    return model



xgb.plot_tree()

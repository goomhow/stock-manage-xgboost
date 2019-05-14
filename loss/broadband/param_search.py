from datetime import datetime
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import *
from sklearn.externals import joblib
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import os
import time


def running_time(func):
    def wrapper(**param):
        startTime = time.time()
        x = func(**param)
        endTime = time.time()
        msecs = int(endTime - startTime)/60
        print("time is %d ms" %msecs)
        return x
    return wrapper


@running_time
def get_transformed_data(fname='bd_train.csv',frac=0.1):
    xName = fname.split(".")[0]+'_x.csv'
    if os.path.exists(xName):
        data = pd.read_csv(xName, index_col='PRD_INST_ID')
    else:
        data = pd.read_csv(fname, index_col='PRD_INST_ID')
        data.sort_index(inplace=True)
        data = data[data.LABEL > -1]
        common = [i[:-2] for i in data.columns if i.endswith('_A')]
        NAME_A = [i+'_A' for i in common]
        NAME_B = [i+'_B' for i in common]
        NAME_C = [i+'_C' for i in common]
        A = data[NAME_A].rename(columns=lambda x: x[:-2])
        B = data[NAME_B].rename(columns=lambda x: x[:-2])
        C = data[NAME_C].rename(columns=lambda x: x[:-2])
        B_A = B - A
        C_B = C - B
        data[NAME_A] = B_A.rename(columns=lambda x: x+'_A')
        data[NAME_B] = C_B.rename(columns=lambda x: x+'_B')
        data.sort_index(inplace=True)
        data.to_csv(xName, index=True, header=True, index_label='PRD_INST_ID')
    d_train = data.sample(frac=frac)
    X = d_train.drop('LABEL', axis=1)
    y = d_train['LABEL']
    return data, X, y


data, X, y = get_transformed_data(frac=0.015)
xgtrain = xgb.DMatrix(X, label=y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
# 第一步 设置XGB params的初始值
params = {'learning_rate': 0.1,
 'n_estimators': 436,
 'max_depth': 8,
 'min_child_weight': 7,
 'gamma': 0.1,
 'subsample': 0.9,
 'colsample_bytree': 0.6,
 'scale_pos_weight': 1,
 'n_jobs': 42,
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


@running_time
def modelfit(useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    alg = XGBClassifier(**params)
    df = data.sample(frac=0.3)
    pX = df.drop('LABEL', axis=1)
    py = df['LABEL']
    if useTrainCV:
        print("start use cv")
        xgb_param = alg.get_xgb_params()
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb_param['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        print(cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
        params['n_estimators'] = cvresult.shape[0]
        print("best tree size is {}".format(cvresult.shape[0]))
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
     'max_depth':[7,8,9],
     'min_child_weight':[7,8,9]
}
grid_search_param(param_grid=param_grid)

# 寻找最优 gamma
param_grid = {'gamma':[i/10.0 for i in range(0, 5)]}
grid_search_param(param_grid=param_grid)

# 寻找最优 scale_pos_weight
param_grid = {'scale_pos_weight': [0, 1, 10, 20]}
grid_search_param(param_grid=param_grid)

#寻找最优 subsample 和 colsample_bytree
param_grid = {
 'subsample': [i/10.0 for i in range(6,10)],
 'colsample_bytree': [i/10.0 for i in range(6,10)]
}
grid_search_param(param_grid=param_grid)

# 寻找最优 reg_alpha
param_grid = {'reg_alpha': [0, 0.005, 0.01, 0.1, 1, 10]}
grid_search_param(param_grid=param_grid)

# 寻找最优 reg_lambda
param_grid = {'reg_lambda': [0, 0.005 ,0.01,0.1,1,10]}
grid_search_param(param_grid=param_grid)

model = XGBClassifier(**params)
model.fit(X, y, eval_metric=metrics.f1_score)


@running_time
def evaluate_model(model_params):
    model = XGBClassifier(**model_params)
    AX = data.drop('LABEL', axis=1)
    ay = data['LABEL']
    X_train, X_test, y_train, y_test = train_test_split(AX, ay, test_size=0.33, random_state=7)
    model.fit(X_train, y_train, eval_metric=metrics.f1_score)
    y_pred = model.predict(X_test)
    accuracy = metrics.acpredict_probacuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('auc:', metrics.roc_auc_score(y_test, y_pred))
    train_report = metrics.classification_report(y_test, y_pred)
    print(train_report)
    feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
    joblib.dump(model, 'lossWarnBroadBandModel_{}.pkl'.format(datetime.now().strftime('%d%H%M')))
    return model


def showPredictInfo(model_name,data):
    model = joblib.load(model_name)
    X_test = data.drop(columns='LABEL')
    y_test = data['LABEL']
    y_pred = model.predict(X_test)
    y_poss = model.predict_proba(X_test)[:,1]
    rdata = pd.DataFrame({
        'y':y_test,
        'y_pred':y_pred,
        'y_poss':y_poss
    })
    print('predict size:%d' % y_pred[y_pred==1].shape[0])
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('auc:', metrics.roc_auc_score(y_test, y_pred))
    train_report = metrics.classification_report(y_test, y_pred)
    print(train_report)
    return rdata

xgb.plot_tree()

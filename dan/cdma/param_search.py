import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import *
from sklearn.model_selection import *
from sklearn.externals import joblib
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


def get_data(month, frac=1):
    data = pd.read_csv('data/dan_{}.csv'.format(month), index_col=ID_COLUMN)
    t = data.sample(frac=frac)
    X = t.drop('LABEL', axis=1)
    y = t['LABEL']
    return data, X, y


def set_best_tree_size(X,y):
    xgtrain = xgb.DMatrix(X, label=y)
    cvresult = xgb.cv(params, xgtrain, num_boost_round=params['n_estimators'], nfold=5, metrics='auc',
                      early_stopping_rounds=50)
    params['n_estimators'] = cvresult.shape[0]
    print('Tree Size is {}'.format(params['n_estimators']))


def grid_search_param(param_grid, X, y):
    gsearch2 = GridSearchCV(estimator=XGBClassifier(**params), param_grid=param_grid, scoring='f1', cv=5)
    gsearch2.fit(X, y)
    print(gsearch2.best_params_, gsearch2.best_score_)
    return gsearch2.best_params_


def evaluate(model_name, fname):
    id = pd.read_csv('../prd_inst_id.csv', index_col='PRD_INST_ID')
    data = pd.read_csv(fname, index_col='PRD_INST_ID').loc[id.index, :].dropna()
    X_test = data.drop(columns='LABEL')
    y_test = data['LABEL']
    return evaluate_part(model_name, X_test, y_test)


def evaluate_part(model, X_test, y_test):
    if isinstance(model,str):
        model = joblib.load(model)
    y_pred = model.predict(X_test)
    y_poss = model.predict_proba(X_test)[:, 1]
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


def update_best_params(params_grid,X,y):
    a = grid_search_param(params_grid,X,y)
    for k, v in params_grid.items():
        i = v.index(a[k])
        if i == 0:
            a[k] = [v[0], (v[1]+v[0])/2.0]
        elif i == len(v)-1:
            a[k] = [(v[-1]+v[-2])/2.0, v[-1]]
        else:
            a[k] = [(v[i - 1] + v[i]) / 2.0, v[i], (v[i] + v[i + 1]) / 2.0]
    print(a)
    params.update(grid_search_param(a,X,y))


def search_params(X, y):
    set_best_tree_size(X, y)
    # 寻找最优 max_depth 和 min_child_weight
    param_grid = {
        'max_depth': list(range(3, 11)),
        'min_child_weight': range(0, 10)
    }
    params.update(grid_search_param(param_grid,X,y))

    # 寻找最优 gamma
    param_grid = {'gamma': [i / 10.0 for i in range(0, 5)]}
    params.update(grid_search_param(param_grid,X,y))

    # 寻找最优 scale_pos_weight
    param_grid = {'scale_pos_weight': [0, 1, 10, 20, 30, 50, 100]}
    params.update(grid_search_param(param_grid,X,y))

    # 寻找最优 subsample 和 colsample_bytree
    param_grid = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    update_best_params(param_grid,X,y)

    # 寻找最优 reg_alpha
    param_grid = {'reg_alpha': [0, 1e-5, 0.001, 0.01, 0.1, 1, 10]}
    update_best_params(param_grid, X, y)

    # 寻找最优 reg_lambda
    param_grid = {'reg_lambda': [0, 1e-5, 0.001, 0.01, 0.1, 1, 10]}
    update_best_params(param_grid, X, y)


if __name__ == '__main__':
    ID_COLUMN = 'PRD_INST_ID'
    LABEL = 'LABEL'
    data, X, y = get_data(month=201806, frac=0.01)
    # 第一步 设置XGB params的初始值
    params = {'learning_rate': 0.1,
              'n_estimators': 2000,
              'max_depth': 6,
              'min_child_weight': 1,
              'gamma': 0,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'scale_pos_weight': 1,
              'n_jobs': 70,
              'objective': 'binary:logistic',
              'reg_alpha': 0.005,
              'reg_lambda': 0.005}
    print("1.第一轮参数调整")
    search_params(X, y)
    print(params)
    print("1.第二轮参数调整")
    search_params(X, y)
    print(params)
    params['n_estimators'] = 2000
    params['learning_rate'] = 0.01
    model = XGBClassifier(**params)
    X_train = data.drop(columns='LABEL')
    y_train = data['LABEL']
    model.fit(X_train, y_train)
    joblib.dump(model, 'model/XGBModel.pkl')
    # 根据模型选择纬度，再进行调参
    sel = SelectFromModel(estimator=model)
    sel.fit(X_train, y_train)
    joblib.dump(sel, 'model/SelectorFromModel.pkl')
    X_train_2 = sel.transform(X_train)
    X_2 = sel.transform(X)
    print("2.第一轮参数调整")
    params = {'learning_rate': 0.1,
              'n_estimators': 2000,
              'max_depth': 6,
              'min_child_weight': 1,
              'gamma': 0,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'scale_pos_weight': 1,
              'n_jobs': 70,
              'objective': 'binary:logistic',
              'reg_alpha': 0.005,
              'reg_lambda': 0.005}
    search_params(X_2, y)
    print(params)
    print("2.第二轮参数调整")
    search_params(X_2, y)
    print(params)
    params['n_estimators'] = 2000
    params['learning_rate'] = 0.01
    model2 = XGBClassifier(**params)
    model2.fit(X_train_2, y_train)
    joblib.dump(model2, 'model/XGBFromModel.pkl')
    del X_train_2
    del X_2
    # 根据纬度得分选择纬度，再进行调参
    sel_best = SelectPercentile(percentile=60)
    sel_best.fit(X_train, y_train)
    joblib.dump(sel, 'model/SelectorPercent.pkl')
    X_train_3 = sel_best.transform(X_train)
    X_3 = sel_best.transform(X)
    print("3.第一轮参数调整")
    params = {'learning_rate': 0.1,
              'n_estimators': 2000,
              'max_depth': 6,
              'min_child_weight': 1,
              'gamma': 0,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'scale_pos_weight': 1,
              'n_jobs': 70,
              'objective': 'binary:logistic',
              'reg_alpha': 0.005,
              'reg_lambda': 0.005}
    search_params(X_2, y)
    print(params)
    print("3.第二轮参数调整")
    search_params(X_2, y)
    print(params)
    params['n_estimators'] = 2000
    params['learning_rate'] = 0.01
    model3 = XGBClassifier(**params)
    model3.fit(X_train_3, y_train)
    joblib.dump(model3, 'model/XGBFromPercent.pkl')
    del data
    del X_train
    del y_train
    del X_train_3
    del X_3
    del y
    data, X_test, y_test = get_data(month=201805)
    del data
    print("\nNORMAL\n")
    evaluate_part(model, X_test, y_test)
    print("\nSELECTOR FROM MODEL\n")
    evaluate_part(model2, X_test, y_test)
    print("\nSELECTOR PERCENT\n")
    evaluate_part(model3, X_test, y_test)

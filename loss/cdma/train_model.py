import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
from xgboost.sklearn import XGBClassifier
import os
import time
from datetime import datetime


def running_time(func):
    def wrapper(*param,**kwargs):
        startTime = time.time()
        x = func(*param,**kwargs)
        endTime = time.time()
        msecs = int(endTime - startTime)/60
        print("time is %d min" %msecs)
        return x
    return wrapper


rm_col = ['BIL_1X_DUR_A',
            'BIL_1X_DUR_B',
            'BIL_1X_DUR_C',
            'CENTREX_FLAG',
            'CHARGE_FT_BEFORE_B',
            'EXACT_FLUX_A',
            'EXACT_FLUX_B',
            'EXACT_FLUX_C',
            'EXTREM_PACK_FLUX_A',
            'EXTREM_PACK_FLUX_B',
            'EXTREM_PACK_FLUX_C',
            'F1X_FLUX_A',
            'F1X_FLUX_B',
            'F1X_FLUX_C',
            'F3G_FLUX_A',
            'F3G_FLUX_B',
            'FIN_OWE_AMT_B',
            'FIN_OWE_AMT_C',
            'HDAY_DAYS_A',
            'HDAY_DUR_A',
            'HDAY_DUR_C',
            'HDAY_FLUX_A',
            'HDAY_FLUX_B',
            'HOME_FLUX_A',
            'IN_EXACT_FLUX_A',
            'IN_EXACT_FLUX_B',
            'IN_PACK_EXACT_FLUX_A',
            'IN_PACK_EXACT_FLUX_B',
            'OFFICE_DUR_A',
            'OFFICE_FLUX_A',
            'OFFICE_FLUX_B',
            'OFF_DUR_A',
            'OFF_FLUX_A',
            'OFF_FLUX_B',
            'OFF_FLUX_C',
            'ON_DUR_A',
            'ON_DUR_B',
            'ON_FLUX_A',
            'ON_FLUX_B',
            'OWE_AMT_A',
            'OWE_AMT_C',
            'OWE_DUR_A',
            'OWE_DUR_B',
            'O_INET_PP_SMS_CNT_A',
            'O_INET_PP_SMS_CNT_B',
            'O_ONET_PP_SMS_CNT_A',
            'O_ONET_PP_SMS_CNT_B',
            'O_ONET_PP_SMS_CNT_C',
            'O_TOL_DSTN_A',
            'O_TOL_DUR_A',
            'O_TOL_DUR_B',
            'PACK_CNT_A',
            'PACK_CNT_B',
            'PACK_CNT_C',
            'PACK_FLAG_A',
            'PACK_FLAG_B',
            'PACK_FLAG_C',
            'PP_SMS_AMT_A',
            'PP_SMS_AMT_B',
            'SP_SMS_AMT_A',
            'SP_SMS_AMT_C',
            'TDD_BIL_DUR_A',
            'TDD_FLUX_A',
            'TDD_FLUX_B',
            'TDD_FLUX_C',
            'TOTAL_1X_CNT_A',
            'TOTAL_1X_CNT_B',
            'TOTAL_1X_CNT_C',
            'TOTAL_3G_CNT_A',
            'TOTAL_3G_CNT_B',
            'TOTAL_FLUX_A',
            'TOTAL_FLUX_B',
            'TOTAL_TDD_CNT_A',
            'T_CALL_DSTN_A',
            'T_ONET_PP_SMS_CNT_A',
            'T_ONET_PP_SMS_CNT_B',
            'T_SP_SMS_CNT_A',
            'T_SP_SMS_CNT_B',
            'VIP_FLAG',
            'WDAY_DUR_A',
            'WDAY_FLUX_A']
ID_COLUMN = 'PRD_INST_ID'
LABEL = 'LABEL'



def get_transformed_data(month='201805'):
    data = pd.read_csv('cdma_{}_x.csv'.format(month), index_col=ID_COLUMN)
    label = pd.read_csv('label/label_{}.csv'.format(month), index_col=ID_COLUMN)
    data[LABEL] = label[label['LABEL'] > -1]
    data.dropna(inplace=True)
    X = data.drop('LABEL', axis=1)
    y = data['LABEL']
    return X, y



def evaluate_model(model_params):
    model = XGBClassifier(**model_params)
    X_train, y_train = get_transformed_data(201805)
    model.fit(X_train, y_train, eval_metric=metrics.f1_score)
    joblib.dump(model, 'lossWarnCdmaModel_{}.pkl'.format(format(datetime.now().strftime('%d%H%M'))))
    del X_train
    del y_train
    X_test, y_test = get_transformed_data(201806)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('auc:', metrics.roc_auc_score(y_test, y_pred))
    train_report = metrics.classification_report(y_test, y_pred)
    print(train_report)
    feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
    print(feat_imp)
    return model


def print_evaluate(model,X_test,y_test):
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('auc:', metrics.roc_auc_score(y_test, y_pred))
    train_report = metrics.classification_report(y_test, y_pred)
    print(train_report)
    feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
    print(feat_imp)


def train_part_full_feature():
    params_a = {'learning_rate': 0.01,
                'n_estimators': 2000,
                'max_depth': 7,
                'min_child_weight': 5,
                'gamma': 0.2,
                'subsample': 0.9,
                'colsample_bytree': 0.6,
                'scale_pos_weight': 9,
                'n_jobs': 42,
                'objective': 'binary:logistic',
                'reg_alpha': 1,
                'reg_lambda': 1e-05}

    params_b = {'learning_rate': 0.01,
                'n_estimators': 2000,
                'max_depth': 9,
                'min_child_weight': 5,
                'gamma': 0.5,
                'subsample': 0.8,
                'colsample_bytree': 0.65,
                'scale_pos_weight': 8,
                'n_jobs': 42,
                'objective': 'binary:logistic',
                'reg_alpha': 1,
                'reg_lambda': 0.005}
    model_a = XGBClassifier(**params_a)
    model_b = XGBClassifier(**params_b)
    X_train, y_train = get_transformed_data(201805)
    model_a.fit(X_train, y_train)
    joblib.dump(model_a, 'AlossWarnCdmaModel_{}.pkl'.format(format(datetime.now().strftime('%d%H%M'))))
    X_train.drop(columns=rm_col, inplace=True)
    model_b.fit(X_train, y_train)
    joblib.dump(model_b, 'BlossWarnCdmaModel_{}.pkl'.format(format(datetime.now().strftime('%d%H%M'))))
    del X_train
    del y_train
    X_test, y_test = get_transformed_data(201806)
    print("*" * 10 + "A" + "*" * 10)
    print_evaluate(model_a, X_test, y_test)
    print("*" * 10 + "B" + "*" * 10)
    print_evaluate(model_b, X_test.drop(columns=rm_col), y_test)


def train_part_feature():
    params = {'learning_rate': 0.01,
                'n_estimators': 2000,
                'max_depth': 9,
                'min_child_weight': 5,
                'gamma': 0.5,
                'subsample': 0.8,
                'colsample_bytree': 0.65,
                'scale_pos_weight': 8,
                'n_jobs': 42,
                'objective': 'binary:logistic',
                'reg_alpha': 1,
                'reg_lambda': 0.005}
    model = XGBClassifier(**params)
    data = pd.read_csv('train_201805_x.csv', index_col='PRD_INST_ID')
    label = pd.read_csv('label/label_201805.csv', index_col='PRD_INST_ID')
    data['LABEL'] = label[label['LABEL'] > -1]
    data.dropna(inplace=True)
    X_train = data.drop(columns='LABEL')
    y_train = data['LABEL']
    model.fit(X_train, y_train)
    joblib.dump(model, 'BlossWarnCdmaModel_{}.pkl'.format(format(datetime.now().strftime('%d%H%M'))))
    del X_train
    del y_train
    del data
    del label
    data = pd.read_csv('cdma_201806_x.csv', index_col='PRD_INST_ID')
    label = pd.read_csv('label/label_201806.csv', index_col='PRD_INST_ID')
    data['LABEL'] = label[label['LABEL'] > -1]
    data.dropna(inplace=True)
    X_test = data.drop(columns='LABEL')
    y_test = data['LABEL']
    del data
    print_evaluate(model, X_test.drop(columns=rm_col), y_test)


if __name__ == '__main__':
    train_part_feature()

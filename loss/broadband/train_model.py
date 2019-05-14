import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
from xgboost.sklearn import XGBClassifier
import os
import time
from datetime import datetime

rm_col= ['ACC_TYPE',
 'ACT_FLAG',
 'AVG_MON_AMT_A',
 'AVG_MON_AMT_B',
 'AVG_MON_AMT_C',
 'BILL_OWE_AMT_A',
 'BILL_OWE_AMT_C',
 'BIL_FLAG',
 'BROADBAND_TICKET_FLAG',
 'CHARGE_FT_BEFORE_B',
 'CYCLE_CHARGE_A',
 'CYCLE_CHARGE_B',
 'DOWN_HDAY_HH15_19_BRD_BND_FLUX_A',
 'DOWN_HDAY_HH15_19_BRD_BND_FLUX_B',
 'DOWN_HDAY_HH15_19_BRD_BND_FLUX_C',
 'DOWN_HDAY_HH19_22_BRD_BND_FLUX_A',
 'DOWN_HDAY_HH19_22_BRD_BND_FLUX_B',
 'DOWN_HDAY_HH19_22_BRD_BND_FLUX_C',
 'DOWN_HDAY_HH9_15_BRD_BND_FLUX_A',
 'DOWN_HDAY_HH9_15_BRD_BND_FLUX_B',
 'DOWN_HDAY_HH9_15_BRD_BND_FLUX_C',
 'DOWN_WDAY_HH15_19_BRD_BND_FLUX_A',
 'DOWN_WDAY_HH15_19_BRD_BND_FLUX_B',
 'DOWN_WDAY_HH15_19_BRD_BND_FLUX_C',
 'DOWN_WDAY_HH19_22_BRD_BND_FLUX_A',
 'DOWN_WDAY_HH19_22_BRD_BND_FLUX_B',
 'DOWN_WDAY_HH19_22_BRD_BND_FLUX_C',
 'DOWN_WDAY_HH9_15_BRD_BND_FLUX_C',
 'FIN_OWE_AMT_A',
 'FIN_OWE_AMT_C',
 'HDAY_BRD_BND_DAYS_A',
 'HDAY_BRD_BND_DAYS_B',
 'HDAY_HH15_19_BRD_BND_CNT_A',
 'HDAY_HH15_19_BRD_BND_CNT_B',
 'HDAY_HH15_19_BRD_BND_CNT_C',
 'HDAY_HH15_19_BRD_BND_DUR_A',
 'HDAY_HH15_19_BRD_BND_DUR_B',
 'HDAY_HH15_19_BRD_BND_DUR_C',
 'HDAY_HH19_22_BRD_BND_CNT_A',
 'HDAY_HH19_22_BRD_BND_CNT_B',
 'HDAY_HH19_22_BRD_BND_CNT_C',
 'HDAY_HH19_22_BRD_BND_DUR_A',
 'HDAY_HH19_22_BRD_BND_DUR_B',
 'HDAY_HH19_22_BRD_BND_DUR_C',
 'HDAY_HH9_15_BRD_BND_CNT_A',
 'HDAY_HH9_15_BRD_BND_CNT_B',
 'HDAY_HH9_15_BRD_BND_CNT_C',
 'HDAY_HH9_15_BRD_BND_DUR_A',
 'HDAY_HH9_15_BRD_BND_DUR_B',
 'HDAY_HH9_15_BRD_BND_DUR_C',
 'OWE_AMT_A',
 'OWE_AMT_B',
 'OWE_AMT_C',
 'OWE_DUR_A',
 'OWE_DUR_B',
 'PAY_FLAG',
 'PRTL_AMT_A',
 'PRTL_AMT_B',
 'PRTL_MONS',
 'REF_TYPE',
 'STOP_DUR',
 'UNIT_CHARGE_A',
 'UP_HDAY_HH15_19_BRD_BND_FLUX_A',
 'UP_HDAY_HH15_19_BRD_BND_FLUX_B',
 'UP_HDAY_HH15_19_BRD_BND_FLUX_C',
 'UP_HDAY_HH19_22_BRD_BND_FLUX_A',
 'UP_HDAY_HH19_22_BRD_BND_FLUX_B',
 'UP_HDAY_HH19_22_BRD_BND_FLUX_C',
 'UP_HDAY_HH9_15_BRD_BND_FLUX_A',
 'UP_HDAY_HH9_15_BRD_BND_FLUX_B',
 'UP_HDAY_HH9_15_BRD_BND_FLUX_C',
 'UP_WDAY_HH15_19_BRD_BND_FLUX_B',
 'UP_WDAY_HH15_19_BRD_BND_FLUX_C',
 'UP_WDAY_HH19_22_BRD_BND_FLUX_A',
 'UP_WDAY_HH19_22_BRD_BND_FLUX_B',
 'UP_WDAY_HH19_22_BRD_BND_FLUX_C',
 'UP_WDAY_HH9_15_BRD_BND_FLUX_C',
 'USER_TYPE_ID',
 'VIP_FLAG',
 'WDAY_HH15_19_BRD_BND_CNT_A',
 'WDAY_HH15_19_BRD_BND_CNT_B',
 'WDAY_HH15_19_BRD_BND_CNT_C',
 'WDAY_HH19_22_BRD_BND_CNT_A',
 'WDAY_HH19_22_BRD_BND_CNT_B',
 'WDAY_HH19_22_BRD_BND_CNT_C',
 'WDAY_HH19_22_BRD_BND_DUR_A',
 'WDAY_HH19_22_BRD_BND_DUR_B',
 'WDAY_HH19_22_BRD_BND_DUR_C',
 'WDAY_HH9_15_BRD_BND_CNT_A',
 'WDAY_HH9_15_BRD_BND_CNT_B']


def running_time(func):
    def wrapper(**param):
        startTime = time.time()
        x = func(**param)
        endTime = time.time()
        msecs = int(endTime - startTime)/60
        print("time is %d min" %msecs)
        return x
    return wrapper

@running_time
def get_transformed_data(fname='bd_train.csv',frac=0.1):
    xName = fname.split(".")[0]+'_x.csv'
    if os.path.exists(xName):
        data = pd.read_csv(xName, index_col='PRD_INST_ID')
    else:
        data = pd.read_csv(fname, index_col='PRD_INST_ID')
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
        data.to_csv(xName, index=True, header=True, index_label='PRD_INST_ID')
    data.sort_index(inplace=True)
    d_train = data.sample(frac=frac)
    X = d_train.drop('LABEL', axis=1)
    y = d_train['LABEL']
    return data, X, y


@running_time
def evaluate_model(model_params):
    model = XGBClassifier(**model_params)
    data, X_train, y_train = get_transformed_data(frac=1)
    model.fit(X_train, y_train, eval_metric=metrics.f1_score)
    joblib.dump(model, 'lossWarnBroadbandModel_{}.pkl'.format(format(datetime.now().strftime('%d%H%M'))))
    del data
    del X_train
    del y_train
    data, X_test, y_test = get_transformed_data(fname='bd_train2.csv', frac=1)
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


if __name__ == '__main__':
    params_a = {'learning_rate': 0.01,
              'n_estimators': 2000,
              'max_depth': 8,
              'min_child_weight': 7,
              'gamma': 0.1,
              'subsample': 0.9,
              'colsample_bytree': 0.6,
              'scale_pos_weight': 3,
              'n_jobs': 42,
              'objective': 'binary:logistic',
              'reg_alpha': 0.2,
              'reg_lambda': 1}
    params_b = {'learning_rate': 0.01,
             'n_estimators': 2000,
             'max_depth': 10,
             'min_child_weight': 8,
             'gamma': 0.1,
             'subsample': 0.9,
             'colsample_bytree': 0.65,
             'scale_pos_weight': 5,
             'n_jobs': 42,
             'objective': 'binary:logistic',
             'reg_alpha': 0.005,
             'reg_lambda': 0.005}

    model_a = XGBClassifier(**params_a)
    model_b = XGBClassifier(**params_b)
    data, X_train, y_train = get_transformed_data(frac=1)
    del data
    model_a.fit(X_train, y_train)
    joblib.dump(model_a, 'AlossWarnBroadbandModel_{}.pkl'.format(format(datetime.now().strftime('%d%H%M'))))
    X_train.drop(columns=rm_col, inplace=True)
    model_b.fit(X_train, y_train)
    joblib.dump(model_b, 'BlossWarnBroadbandModel_{}.pkl'.format(format(datetime.now().strftime('%d%H%M'))))
    del X_train
    del y_train
    data, X_test, y_test = get_transformed_data(fname='broadband_201806.csv', frac=1)
    del data
    print("*"*10+"A"+"*"*10)
    print_evaluate(model_a, X_test, y_test)
    print("*" * 10 + "B" + "*" * 10)
    print_evaluate(model_b, X_test.drop(columns=rm_col), y_test)
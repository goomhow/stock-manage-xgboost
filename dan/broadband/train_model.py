import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
from xgboost.sklearn import XGBClassifier
import os
import time
from datetime import datetime
ID_COLUMN = 'PRD_INST_ID'


def running_time(func):
    def wrapper(*param,**kwargs):
        startTime = time.time()
        x = func(*param,**kwargs)
        endTime = time.time()
        msecs = int(endTime - startTime)/60
        print("time is %d min" %msecs)
        return x
    return wrapper


rm_col = rm_col = ['BIL_1X_DUR_A',
 'CENTREX_FLAG',
 'CHARGE_2809_A',
 'CHARGE_BEFORE_A',
 'CHARGE_FT_BEFORE_A',
 'CHARGE_FT_BEFORE_B',
 'CHARGE_FT_BEFORE_C',
 'COUNTRY_FLAG',
 'CUST_TYPE_ID',
 'EXACT_FLUX_A',
 'EXACT_FLUX_B',
 'EXACT_FLUX_C',
 'EXTREM_BASE_FLUX_A',
 'EXTREM_FLUX_A',
 'EXTREM_FLUX_B',
 'EXTREM_PACK_FLUX_A',
 'EXTREM_PACK_FLUX_B',
 'EXTREM_PACK_FLUX_C',
 'F1X_FLUX_A',
 'F1X_FLUX_B',
 'F3G_FLUX_C',
 'GENDER_ID',
 'HDAY_DAYS_B',
 'HDAY_FLUX_C',
 'HOME_DUR_B',
 'HOME_DUR_C',
 'HOME_FLUX_A',
 'HOME_FLUX_C',
 'IN_BASE_EXACT_FLUX_B',
 'IN_BASE_EXACT_FLUX_C',
 'IN_EXACT_FLUX_A',
 'IN_EXACT_FLUX_B',
 'IN_EXACT_FLUX_C',
 'IN_PACK_EXACT_FLUX_A',
 'IN_PACK_EXACT_FLUX_B',
 'IN_PACK_EXACT_FLUX_C',
 'LATN_ID',
 'OFFICE_DUR_A',
 'OFFICE_DUR_B',
 'OFFICE_DUR_C',
 'OFFICE_FLUX_A',
 'OFFICE_FLUX_B',
 'OFFICE_FLUX_C',
 'OFF_DUR_B',
 'OFF_DUR_C',
 'OFF_FLUX_C',
 'OLD_PRD_INST_TYPE_ID',
 'ON_DUR_A',
 'ON_DUR_C',
 'OUT_EXACT_FLUX_A',
 'OUT_EXACT_FLUX_B',
 'OUT_EXACT_FLUX_C',
 'O_INET_PP_SMS_CNT_A',
 'O_INET_PP_SMS_CNT_B',
 'O_INET_PP_SMS_CNT_C',
 'O_ONET_PP_SMS_CNT_A',
 'O_ONET_PP_SMS_CNT_B',
 'O_ONET_PP_SMS_CNT_C',
 'O_TOL_CNT_A',
 'O_TOL_CNT_C',
 'O_TOL_DSTN_A',
 'PACK_CNT_A',
 'PACK_CNT_B',
 'PACK_CNT_C',
 'PACK_FLAG_A',
 'PACK_FLAG_C',
 'PAY_MODE_ID',
 'PP_SMS_AMT_B',
 'PP_SMS_AMT_C',
 'SP_SMS_AMT_A',
 'SP_SMS_AMT_B',
 'SP_SMS_AMT_C',
 'STD_PRD_INST_STAT_ID_A',
 'STD_PRD_INST_STAT_ID_B',
 'STD_PRD_INST_STAT_ID_C',
 'TDD_BIL_DUR_C',
 'TDD_FLUX_A',
 'TDD_FLUX_B',
 'TDD_FLUX_C',
 'TOTAL_1X_CNT_B',
 'TOTAL_1X_CNT_C',
 'TOTAL_3G_CNT_B',
 'TOTAL_3G_CNT_C',
 'TOTAL_FLUX_A',
 'TOTAL_FLUX_B',
 'TOTAL_TDD_CNT_A',
 'TOTAL_TDD_CNT_B',
 'T_ONET_PP_SMS_CNT_B',
 'T_ONET_PP_SMS_CNT_C',
 'T_SP_SMS_CNT_A',
 'T_SP_SMS_CNT_B',
 'T_SP_SMS_CNT_C',
 'USER_TYPE_ID',
 'WDAY_DAYS_A',
 'WDAY_DAYS_B',
 'WDAY_DAYS_C',
 'WDAY_DUR_A',
 'WDAY_DUR_C',
 'WDAY_FLUX_A',
 'WDAY_FLUX_C']

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


@running_time
def evaluate_model(model_params):
    model = XGBClassifier(**model_params)
    data, X_train, y_train = get_transformed_data(frac=1)
    model.fit(X_train, y_train, eval_metric=metrics.f1_score)
    joblib.dump(model, 'danCdmaModel_{}.pkl'.format(format(datetime.now().strftime('%d%H%M'))))
    del data
    del X_train
    del y_train
    data, X_test, y_test = get_transformed_data(fname='cdma_train2.csv', frac=1)
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
    print('predict size:%d' % y_pred[y_pred == 1].shape[0])
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('auc:', metrics.roc_auc_score(y_test, y_pred))
    train_report = metrics.classification_report(y_test, y_pred)
    print(train_report)
    feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
    print(feat_imp)

def trainFullAndPartFeature():
    params_a = {'learning_rate': 0.01,
     'n_estimators': 1000,
     'max_depth': 5,
     'min_child_weight': 4,
     'gamma': 0,
     'subsample': 0.9,
     'colsample_bytree': 0.7,
     'scale_pos_weight': 20,
     'n_jobs': 50,
     'objective': 'binary:logistic',
     'reg_alpha': 0.1,
     'reg_lambda': 0.005}


    params_b = {'learning_rate': 0.01,
     'n_estimators': 1000,
     'max_depth': 6,
     'min_child_weight': 1,
     'gamma': 0.0,
     'subsample': 0.8,
     'colsample_bytree': 0.8,
     'scale_pos_weight': 22,
     'n_jobs': 50,
     'objective': 'binary:logistic',
     'reg_alpha': 1,
     'reg_lambda': 0.005}
    model_a = XGBClassifier(**params_a)
    model_b = XGBClassifier(**params_b)
    data, X_train, y_train = get_transformed_data(month='201805', frac=1)
    del data
    model_a.fit(X_train, y_train)
    joblib.dump(model_a, 'ADanBDModel_{}.pkl'.format(format(datetime.now().strftime('%d%H%M'))))
    X_train.drop(columns=rm_col, inplace=True)
    model_b.fit(X_train, y_train)
    joblib.dump(model_b, 'BDanBDModel_{}.pkl'.format(format(datetime.now().strftime('%d%H%M'))))
    del X_train
    del y_train
    data, X_test, y_test = get_transformed_data(month='201806', frac=1)
    del data
    print("*"*10+"A"+"*"*10)
    print_evaluate(model_a, X_test, y_test)
    print("*" * 10 + "B" + "*" * 10)
    print_evaluate(model_b, X_test.drop(columns=rm_col), y_test)


def sampleTrain():
    LABEL = 'LABEL'
    dpath = 'dan_train_{}.csv'
    data4 = pd.read_csv(dpath.format(201804),index_col=ID_COLUMN)
    data5 = pd.read_csv(dpath.format(201805),index_col=ID_COLUMN)
    a1 = data4[data4.LABEL == 1]
    del data4
    b1 = data5[data5.LABEL == 1]
    b0 = data5[data5.LABEL == 1].sample(n=(a1.shape[0]+b1.shape[0])*35)
    del data5
    data = b0.append(a1).append(b1).sort_index()
    X = data.drop(columns=LABEL)
    y = data[LABEL]
    params = {'learning_rate': 0.01,
     'n_estimators': 1000,
     'max_depth': 8,
     'min_child_weight': 0,
     'gamma': 0.4,
     'subsample': 0.9,
     'colsample_bytree': 0.6,
     'scale_pos_weight': 10,
     'n_jobs': 50,
     'objective': 'binary:logistic',
     'reg_alpha': 1,
     'reg_lambda': 1}
    model = XGBClassifier(**params)
    model.fit(X, y, eval_metric=metrics.f1_score)
    del X
    del y
    del data
    joblib.dump(model, 'CDanCdmaModel_{}.pkl'.format(format(datetime.now().strftime('%d%H%M'))))
    data, X_test, y_test = get_transformed_data(month='201806', frac=1)
    print_evaluate(model, X_test, y_test)


if __name__ == '__main__':
    sampleTrain()

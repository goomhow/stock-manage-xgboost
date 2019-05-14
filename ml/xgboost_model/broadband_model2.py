from datetime import datetime
import numpy as np
import pandas as pd
from sklearn import metrics, learning_curve, svm
from sklearn.model_selection import *
from sklearn.externals import joblib
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.plotting import plot_importance
import matplotlib.pyplot as plt

def single_model_xgb(X,y,**params):
    flag = "LABEL"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)  # 94.62%
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('auc:', metrics.roc_auc_score(y_test, y_pred))
    print('精确率Accuracy:', metrics.accuracy_score(y_test, y_pred))
    train_report = metrics.classification_report(y_test, y_pred)
    print(train_report)
    return model


def learning_curve_fun(model, X, y, title=None, cv=None):
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    title = "xgboost learning rate"
    plot_learning_curve(model, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


def svm_model(d_train, flag="label"):
    from sklearn.preprocessing import MinMaxScaler, Normalizer
    normal = Normalizer()
    scaler = MinMaxScaler()
    X = d_train.drop(flag, axis=1, inplace=False)
    y = d_train[flag]
    X = normal.fit(X, y).transform(X)
    X = scaler.fit(X, y).transform(X)
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = svm.SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('auc:', metrics.roc_auc_score(y_test, y_pred))
    print('精确率Accuracy:', metrics.accuracy_score(y_test, y_pred))
    train_report = metrics.classification_report(y_test, y_pred)
    print(train_report)
    return model


def param_search(d_train, flag='label'):
    X = d_train.drop(flag, axis=1).drop('PRD_INST_ID', axis=1)
    y = d_train[flag]
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    tuned_parameters = [{'max_depth': [5, 6, 7],
                         'n_estimators':[2000, 3000],
                         'scale_pos_weight':[20, 50, 200],
                         'subsample':[0.75, 0.85, 0.95],
                         'colsample_bytree':[0.75, 0.85, 0.95],
                         }]
    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(XGBClassifier(
                            learning_rate=0.01,
                            min_child_weight=3,
                            gamma=0.3,
                            objective='binary:logistic',
                            seed=27,
                            nthread=12,
                            reg_alpha=0.01),
                           tuned_parameters,
                           cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(metrics.classification_report(y_true, y_pred))
        print()


def parse(data):
    f0 = ['INV_AMT_ALL', 'PRO_I_DUR_ALL', 'PRO_FLUX_ALL','THSC','YWQF','CWQF']
    f2 = [i for i in data.columns if i[-1] == '2']
    f3 = [i for i in data.columns if i[-1] == '3']
    f0.extend(f2)
    f0.extend(f3)
    for k in f0:
        data[k] = pd.cut(data[k], 7, labels=list(range(0, 7))).astype(np.int16)
    return data


def predict_df(model, df_test='', label='label', pfname="broadband_predict.csv"):
    if not isinstance(df_test, pd.DataFrame):
        print('*'*10+'LOAD DATA'+'*'*10)
        df_test = pd.read_csv(pfname)
        df_test = df_test[df_test[label] == 0]
    print('*' * 10 + 'LOAD MODEL' + '*' * 10)
    bst = joblib.load(model)
    X_test = df_test.drop(label, axis=1).drop("PRD_INST_ID", axis=1)
    y_pred = bst.predict(X_test)
    y_test_pred1 = bst.predict_proba(X_test)
    out = pd.DataFrame(
        {
            'LATN_ID': df_test["LATN_ID"].astype(np.int64),
            'PRD_INST_ID': df_test["PRD_INST_ID"].astype(np.int64),
            'PREDICT': y_pred.astype(np.int32),
            'POSSIBILITY': y_test_pred1[:, 1].astype(np.float)
        }
    )
    out = out[out['PREDICT'] == 1].sort_values(by=['PREDICT', 'POSSIBILITY'], ascending=False)
    out.to_csv(generate_filename("broadband_result{}.csv"), header=True, index=None, )
    return out


def generatorModel(data_file="broadband_train.csv", save_file=None):
    data = pd.read_csv(data_file)
    model = single_model_xgb(data)
    if not save_file:
        save_file = generate_filename("broadband_xgb_{}.pkl")
    joblib.dump(model, save_file)
    return save_file


def mixData(fmodel,label='label', pfname="broadband_predict.csv"):
    print('*' * 10 + 'LOAD DATA' + '*' * 10)
    df_test = pd.read_csv(pfname)
    df_test = df_test[df_test[label] == 0]
    precision_out = predict_df(fmodel, df_test)
    recall_out = predict_df(f, df_test)
    ySet = set(precision_out['PRD_INST_ID'])
    another = recall_out.loc[lambda x:x.PRD_INST_ID not in ySet, :]
    append_size = 15000 - precision_out.shape[0]
    result = recall_out.append(another.iloc[:append_size, :])
    result.to_csv(generate_filename("broadband_mix{}.csv"), header=True, index=None)
    return result


def generate_filename(fmt):
    date = datetime.now()
    date_str = date.strftime('%m%d%H%M')
    return fmt.format(date_str)

d_train = pd.read_csv('broadband_train.csv')
X = d_train.drop('LABEL', axis=1).drop("PRD_INST_ID", axis=1)
y = d_train['LABEL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
params = {'learning_rate':0.03,'n_estimators':5000,'max_depth':6,'min_child_weight':3,'gamma':0.3,
          'subsample':0.8,'colsample_bytree':0.8,'scale_pos_weight':300,'nthread':16,
          'reg_alpha':0.005,'reg_lambda': 0.01}


xgtrain = xgb.DMatrix(X, label=y)
cvresult = xgb.cv(params, xgtrain, num_boost_round=1000, nfold=5,metrics='auc', early_stopping_rounds=50)

def modelfit(alg, X = X,y = y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X, label=y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(X, y, eval_metric='auc')
    # Predict training set:
    dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:, 1]
    # Print model report:
    print
    "\nModel Report"
    print
    "Accuracy : %.4g" % metrics.accuracy_score(X, dtrain_predictions)
    print
    "AUC Score (Train): %f" % metrics.roc_auc_score(X, dtrain_predprob)
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    print(feat_imp)

# def xgb_model(**kwargs):
#     model = xgb.XGBClassifier(**kwargs)
#     model.fit(X_train, y_train)  # 94.62%
#     y_pred = model.predict(X_test)
#     accuracy = metrics.accuracy_score(y_test, y_pred)
#     print("Accuracy: %.2f%%" % (accuracy * 100.0))
#     print('auc:', metrics.roc_auc_score(y_test, y_pred))
#     print('PRECISION:', metrics.accuracy_score(y_test, y_pred))
#     print('RECALL:', metrics.recall_score(y_test, y_pred))
#     train_report = metrics.classification_report(y_test, y_pred)
#     print(train_report)
#     return model
#
#
# def show_model(fmodel):
#     model = joblib.load(fmodel)
#     y_pred = model.predict(X_test)
#     accuracy = metrics.accuracy_score(y_test, y_pred)
#     print('*'*10+fmodel+'*'*10)
#     print("Accuracy: %.2f%%" % (accuracy * 100.0))
#     print('auc:', metrics.roc_auc_score(y_test, y_pred))
#     print('精确率Accuracy:', metrics.accuracy_score(y_test, y_pred))
#     train_report = metrics.classification_report(y_test, y_pred)
#     print(train_report)
#     return model


#broadband_xgb_06131154.pkl  broadband_xgb_06131114.pkl
if __name__ == '__main__':
    start = datetime.now()
    print(mixData('broadband_xgb_06131154.pkl'))
    print(mixData('broadband_xgb_06131114.pkl'))
    end = datetime.now()
    print("model train cost {}s".format((end-start).seconds))
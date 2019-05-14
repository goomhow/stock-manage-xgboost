import pandas as pd
from sklearn import metrics
from sklearn.model_selection import *
import xgboost as xgb
import matplotlib.pyplot as plt

flag = "label"
d_train = pd.read_csv('broadband_train.csv').sample(frac=0.5)
X = d_train.drop(flag, axis=1).drop("PRD_INST_ID", axis=1)
y = d_train[flag]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)


def modelfit(model, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train.values, label=y_train.values)
        xgtest = xgb.DMatrix(X_test.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'],
                          nfold=5,metrics='auc', early_stopping_rounds=50)
        model.set_params(n_estimators=cvresult.shape[0])

    # Fit the modelorithm on the data
    model.fit(X_train, eval_metric='auc')

    # Predict training set:
    dtrain_predictions = model.predict(X_train)
    dtrain_predprob = model.predict_proba(X_train)[:, 1]

    # Print model report:
    print("\nTrainSet Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))

    test_predictions = model.predict(X_test)
    test_predprob = model.predict_proba(X_test)[:, 1]
    #Predict on testing data:
    print("\nTestSet Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, test_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, test_predprob))
    feat_imp = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


def single_model_xgb():
    model = xgb.XGBClassifier(learning_rate=0.01,
                              n_estimators=1450,
                              max_depth=8,
                              min_child_weight=310,
                              gamma=0.5,
                              subsample=0.65,
                              colsample_bytree=0.95,
                              objective='binary:logistic',
                              scale_pos_weight=200,
                              seed=27,
                              nthread=12,
                              reg_alpha=1,
                              reg_lambda=0)

    model.fit(X_train, y_train)  # 94.62%
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print('auc:', metrics.roc_auc_score(y_test, y_pred))
    print('精确率Accuracy:', metrics.accuracy_score(y_test, y_pred))
    train_report = metrics.classification_report(y_test, y_pred)
    print(train_report)
    return model


param_test1 = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}


def search_params(params):
    model = xgb.XGBClassifier(learning_rate=0.01,
                              n_estimators=1450,
                              max_depth=8,
                              min_child_weight=310,
                              gamma=0.5,
                              subsample=0.65,
                              colsample_bytree=0.95,
                              objective='binary:logistic',
                              scale_pos_weight=200,
                              seed=27,
                              nthread=12,
                              reg_alpha=1,
                              reg_lambda=0)
    grid = GridSearchCV(estimator=model, param_grid=params,
                        scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    grid.fit(X, y)
    print('best_score_:')
    print(grid.best_score_)
    print('best_params_')
    print(grid.best_params_)

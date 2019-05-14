import gc
import time
from datetime import datetime

from functools import partial

from heamylab import mini_sample
import pandas as pd
import numpy as np

# import lightgbm as lgb
# from lightgbm.plotting import plot_importance, plot_metric, plot_tree, create_tree_digraph

import xgboost as xgb
from sklearn import metrics
from xgboost import XGBClassifier
from xgboost.plotting import plot_importance, plot_tree, to_graphviz

from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, learning_curve, ShuffleSplit
from sklearn.model_selection import train_test_split  # 训练集数据拆分
from sklearn.metrics import (roc_curve, auc, roc_auc_score, accuracy_score, precision_recall_fscore_support,
                             classification_report)  # 模型评估
from sklearn.ensemble import (GradientBoostingClassifier, VotingClassifier,
                              BaggingClassifier, BaggingRegressor, RandomForestClassifier)

from sklearn.externals import joblib
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

def load_data_source(trainf, testf=None):
    return pd.read_csv(trainf, sep=","),  pd.read_csv(testf, sep=",")

def load_data(trainf, testf=None):
    train = pd.read_csv(trainf, sep=",")
    test = pd.read_csv(testf, sep=",")
    train["flag"] = 0
    test["flag"] = 1

    temp = pd.concat([train, test])

    temp.drop("MERGE_PROM_INST_ID", axis=1, inplace=True)
    f0 = ['INV_AMT_ALL', 'PRO_I_DUR_ALL', 'PRO_FLUX_ALL']
    f2 = [i for i in train.columns if i[-1] in ['2', '3']]
    f0.extend(f2)

    for k in f0:
        temp[k] = pd.cut(temp[k], 7, labels=list(range(0, 7))).astype(np.int16)


    """
    THSC:2709
    YWQF:1035  Binarization
    CWQF:919   
    """


    # X = Normalizer().fit_transform(StandardScaler().fit_transform(temp[[x for x in temp.columns if x not in ["flag", flag]]]))

    return temp.query("flag == 0").iloc[:, :-1], temp.query("flag == 1").iloc[:, :-1]


def data_preprocess(df):
    pass


def single_model_lr(df):
    pass


flag = "label"
time_time = int(time.time() * 1000)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
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
    plt.savefig('learning_rate.png')


# def single_model_lgb(train):
#     X = train[[x for x in train.columns if x not in [flag]]]
#     Y = train[flag]
#
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
#     cv_params = {
#         # 'n_estimators': range(60, 75, 5),
#         # 'num_leaves': range(70, 110, 5),
#         # 'min_data_in_leaf': range(2, 5, 1),
#         # 'subsample': [0.5, 0.6, 0.7, 0.75, 0.85, 0.9],
#         # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
#         # 'min_split_gain': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
#         # 'reg_alpha': [2, 3, 4, 5],
#         # 'reg_lambda': [2, 3, 4, 5],
#         # 'learning_rate': [0.01, 0.05, 0.1]
#
#         # 'n_estimators': range(1000, 1300, 100),
#         # 'num_leaves': range(80, 200, 20),
#         # 'min_data_in_leaf': range(2, 5, 1),
#         # 'subsample': [0.7, 0.75, 0.85, 0.9, 0.95],
#         # 'colsample_bytree': [0.7, 0.8, 0.9, 0.95],
#         # 'min_split_gain': [0.1, 0.2],
#         # 'reg_alpha': [2, 3, 4, 5],
#         # 'reg_lambda': [2, 3, 4, 5],
#         # 'learning_rate': [0.01, 0.05, 0.1]
#     }
#     optimized_GBM = lgb.LGBMClassifier(boosting_type='gbdt',
#                                        learning_rate=0.1,
#                                        n_estimators=60, num_leaves=70,
#                                        max_depth=-1,
#                                        min_data_in_leaf=5,
#                                        subsample=0.85, colsample_bytree=0.9,
#                                        min_split_gain=0.3,
#                                        reg_alpha=2, reg_lambda=5,
#                                        metric='roc_auc')
#
#     # optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_absolute_error', cv=5,
#     #                              verbose=1, n_jobs=4)
#     optimized_GBM.fit(x_train, y_train)
#     evalute_result = optimized_GBM.evals_result_
#     print('每轮迭代运行结果:{0}'.format(evalute_result))
#     # print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
#     print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
#     lgb_name = 'lgb_{}.pkl'.format(int(time.time() * 1000))
#     # joblib.dump(optimized_GBM, lgb_name)
#
#     ### 特征选择
#     # df1 = pd.DataFrame(X.columns.tolist(), columns=['feature'])
#     # df1['importance'] = list(optimized_GBM.feature_importance())
#     # df1 = df1.sort_values(by='importance', ascending=False)
#     # df1.to_csv("feature_score_20180530.csv", index=None, )
#
#     return lgb_name


def single_model_xgb(d_train, show_learning=False):
    X = d_train.drop(flag,axis=1).drop("PRD_INST_ID",axis=1)
    y = d_train[flag]

    sm = SMOTE(random_state=5)
    X, y = sm.fit_sample(X, y)
    #X = pd.DataFrame(X, columns=[[x for x in d_train.columns if x not in [flag,""]]])

    seed = 7
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    base_para = {'n_estimators': 2500,
                 'learning_rate': 0.03,
                 'max_depth': 7,
                 'seed': 2018,
                 'metrics': 'pr',
                 'min_child_weight': 6,
                 'gamma': 0.55,
                 'subsample': 0.65,
                 'colsample_bytree': 0.95,
                 'reg_alpha': 3,
                 'reg_lambda': 4,
                 'n_jobs': 16}

    model = xgb.XGBClassifier(**base_para)

    model.fit(X_train, y_train)  # 94.62%

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # print("1-{} 2-{} 3-{}".format(model.best_score, model.best_iteration, model.best_ntree_limit))
    # print(plot_importance(model, max_num_features=20))
    # plt.show()
    # plt.savefig("fearture.png")
    print('auc:', roc_auc_score(y_test, y_pred))
    print('精确率Accuracy:', accuracy_score(y_test, y_pred))
    train_report = metrics.classification_report(y_test, y_pred)
    print(train_report)

    xgb_name = 'xgb_{}.pkl'.format(int(time.time() * 1000))
    joblib.dump(model, xgb_name)

    if show_learning:
        learning_curve_fun(model, X, y)

    return xgb_name


def learning_curve_fun(model, X, y, title=None, cv=None):
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    title = "xgboost learning rate"
    plot_learning_curve(model, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)


def metric_model(model):
    pass


def ensemble_vote(df):
    pass


def ensemble_stacking(df):
    pass


def predict_df(df_test, model=None, test_type=False, out_flag=0):
    print('加载模型')
    bst = joblib.load(model)

    X_test = df_test.drop("label",axis=1).drop("PRD_INST_ID",axis=1)
    print(X_test.shape, list(X_test.columns))
    y_test = df_test[flag]

    y_pred = bst.predict(X_test)
    y_test_pred1 = bst.predict_proba(X_test)

    out = pd.DataFrame(
        {
            'Prd_Inst_Id': df_test["PRD_INST_ID"].astype(np.int64),
            'flag': y_pred.astype(np.int32),
            'flag1': y_test_pred1[:, 1].astype(np.float)
        })

    if test_type:
        train_report = classification_report(y_test, y_pred)
        print(train_report)
        out["t"] = df_test["label"].as_matrix()
        print("真实流失数", out.query("t > 0").shape)
        print("真实命中数", out.query("flag > 0 and t > 0").shape)
        print("分段命中数", out.query("flag1 > 0.75 and t > 0").shape)
        print("分段命中数", out.query("flag1 > 0.75").shape)
        print('auc:', roc_auc_score(out["t"], y_pred))

    outf = "cdma_{}.csv".format(time_time)
    #out = out.sort_values('flag1', ascending=False)

    print("预测流失数", out.query("flag > 0").shape)

    if out_flag:
        out.to_csv(outf, header=True, index=None, )
   
   #     out_data("cdma_predict_new.txt",
    #             outf)
    #
        return outf


def out_data(source_file, predict_file, keys=None, type=None):
    df = load_data_source(source_file)
    # df.rename(columns={"Prd_Inst_Id_a": "Prd_Inst_Id"}, inplace=True)

    df1 = pd.read_csv(predict_file)

    out = pd.merge(df1, df, left_on="Prd_Inst_Id", right_on="Prd_Inst_Id_a")
    print(out.head())

    if type:
        print(out.query("flag == t and t > 0").shape)
        print(out.query("flag1>.65 and t >0").shape)

    out.to_csv("ourt_{}.csv".format(time_time), header=True)


def daikuan_proc(dk):
    if "K" in dk:
        return 5
    if "G" in dk or "未知速率" in dk:
        return 120

    dk = int(dk[:-1])

    big_cust = [1, 2, 4, 8, 10, 20, 50, 100]
    if dk in big_cust:
        return dk
    elif dk < 10:
        return 5
    elif dk < 20:
        return 15
    elif dk < 50:
        return 45
    elif dk < 100:
        return 80
    else:
        return 120


def age_proc(data):
    if data == 0:
        return 0
    if 18 <= data < 30:
        return 1
    elif 30 <= data < 50:
        return 2
    elif 50 <= data < 70:
        return 3
    else:
        return 4


def single_RandomForest(d_train, d_test):
    clf = RandomForestClassifier()

    X = d_train[[x for x in d_train.columns if x not in [id, flag]]]
    y = d_train[flag]

    test_d = d_test[[x for x in d_test.columns if x not in [id, flag]]]

    clf.fit(X, y)

    y_pred = clf.predict(test_d)

    dd = pd.DataFrame({"t": y_pred.astype(np.int32)})
    print(dd.query("t > 0").shape)


# def mini_sample(d_train, df_test):
#     X_train = d_train[[x for x in d_train.columns if x not in [id, flag]]]
#     y_train = d_train[flag]
#
#     X_test = df_test[[x for x in df_test.columns if x not in [id, flag]]]
#     y_test = df_test[flag]
#
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=111)
#     # 实际场景下，自行准备
#
#     # create dataset
#     dataset = Dataset(X_train, y_train, X_test)
#
#     from sklearn.svm import SVC
#
#     # initialize RandomForest & LinearRegression
#     model_rf = Regressor(dataset=dataset, estimator=RandomForestRegressor, parameters={'n_estimators': 50}, name='rf')
#     model_lr = Regressor(dataset=dataset, estimator=LinearRegression, parameters={'normalize': True}, name='lr')
#     xgbclf = Classifier(dataset=dataset, estimator=XGBClassifier,
#                         parameters={"n_estimators": 110, "max_depth": 5, "min_child_weight": 3, "subsample": .95,
#                                     "colsample_bytree": .95, "learning_rate": .05}, name='xgb')
#
#     rf_model = Classifier(dataset=dataset, estimator=RandomForestClassifier, name="rft")
#     adb_model = Classifier(dataset=dataset, estimator=AdaBoostClassifier, name="adb")
#     gdbc_model = Classifier(dataset=dataset, estimator=GradientBoostingClassifier, name="gbdt")
#     et_model = Classifier(dataset=dataset, estimator=ExtraTreesClassifier, name="et")
#
#     # Stack two models
#     # Returns new dataset with out-of-fold predictions
#     pipeline = ModelsPipeline(model_rf, model_lr, xgbclf, rf_model, adb_model, gdbc_model, et_model)  # 594
#     stack_ds = pipeline.stack(k=10, seed=111)
#
#     # Train LinearRegression on stacked data (second stage)
#     stacker = Regressor(dataset=stack_ds, estimator=LinearRegression)
#     results = stacker.predict()
#
#     print("out:",
#           pd.DataFrame({'Prd_Inst_Id': df_test["Prd_Inst_Id_a"].as_matrix(), "flag": results}).query("flag > .5").shape)
#     # Validate results using 10 fold cross-validation
#     results = stacker.validate(k=10, scorer=mean_absolute_error)


"""
    model_rf = Regressor(dataset=dataset, estimator=RandomForestRegressor, parameters={'n_estimators': 151}, name='rf')
    model_lr = Regressor(dataset=dataset, estimator=LinearRegression, parameters={'normalize': True}, name='lr')
    model_knn = Regressor(dataset=dataset, estimator=KNeighborsRegressor, parameters={'n_neighbors': 15}, name='knn')

    pipeline = ModelsPipeline(model_rf, model_lr, model_knn)
    stack_ds = pipeline.stack(k=5, seed=111)

    # 2nd level
    stack_rf = Regressor(dataset=stack_ds, estimator=RandomForestRegressor, parameters={'n_estimators': 15}, name='rf')
    stack_lr = Regressor(dataset=stack_ds, estimator=LinearRegression, parameters={'normalize': True}, name='lr')
    stack_pipeline = ModelsPipeline(stack_rf, stack_lr)

    # 3rd level
    weights = stack_pipeline.find_weights(mean_absolute_error)
    print('---')
    result = stack_pipeline.weight(weights).validate(mean_absolute_error, 10)
"""


def pu_learning(df_train_data_org, df_test_data_org, df_train, df_test):
    """
    SPY算法
    Spy 的基本思想是从 P 中划分出一个子集 S，将 S 中的样本放到 U 中，从而得到新的正样本集 P-S 和未标识样本集 U+S。使用 P-S 作为正样本，U+S 作为负样本，
    利用迭代的 EM 算法进行分类，当分类结束后，利用对那些「间谍」样本的标识，确定一个参数阈值 th，再对 U 中的文档进行划分得到可靠的反样本集合 RN。
    其中，从 P 中划分子集 S 的数量比例一般为 15%。算法步骤描述如下：
    （1）RN 集合置空；
    （2）从 P 中随机选取子集 S，得到新的正样本集 PS=P-S 和未标识样本集 US=U+S，记 PS 中各个样本类别为 1，US 各个样本类别为-1；
    （3）PS 和 US 作为训练集，用 I-EM 算法训练得到一个贝叶斯分类器；
    （4）使用子集 S 确定出一个概率阈值 th；
    （5）对 US 中的每个样本 d 使用贝叶斯分类器计算其属于正类别的概率 P(1|d)，如果小于阈值概率 th，则把其加入 RN 集合。

    SPY 是刘冰等人为解决 PU-learning 中只有正样本和无标签样本，而没有负样本问题，在 ICML 2002 上提出的一种从无标签样本中寻找最可能的负例，
    以便有监督学习的技术。在比赛过程中，我们借鉴其方法，用来从源领域 Ds 中筛选和目标领域 Dt 最相似的样本，然后将选出的样本和 Dt 中的样本放在
    一起训练模型。SPY 的算法流程为：

    样本重标记：将 Ds 和 Dt 对应的数据重新标记，即 Ds 中的样本标记为负样本，Dt 中的样本标记为正样本；
    数据集切分：将重新标记后的 Ds 和 Dt 放在一起，并随机切分为训练集、测试集；
    分类器训练：选择一个可以输出概率预测结果的基分类器，并在训练集上训练该分类器；
    概率预测：将训练好的模型在测试集上预测，输出测试集中每条样本属于正样本的概率；
    样本筛选：将测试集中样本按概率降序排序，设置阈值 k，记不小于第 k 个正样本对应的概率为 p，选择 Ds 中概率不小于 p 的样本，放入 Dt 中。

    从上述流程中可以看到，我们首先利用 Dt 和 Ds 中的样本训练一个“裁判”，对测试集中样本打分。然后，利用测试集合中属于 Dt 的样本作为“内应”，
    根据“裁判”的评判结果，选取出 Ds 中，比 Dt 的部分样本，更“像”Dt 的样本。

    在应用 SPY 方法时，可能有两点需要注意：基分类器的选择和筛选阈值的设置。若基分类器太强，SPY 第一步训练出的分类模型效果很好，将不能从 Ds
    中选择出足够数量的样本，因此我们选择了 LR 和 NB（朴素贝叶斯）作为基分类器，而非 GBM。另外，筛选阈值的设置需要根据实验结果调整，尽可能既
    不错杀，也不误放。若设置的过高，会使得和 Dt 差别较大的样本被选中，反之则会遗漏和 Dt 较相似的样本。另外，SPY 可以和交叉验证、集成学习相结
    合，提高选择出的样本置信度，或增加样本的选择数量。
    """

    from sklearn.preprocessing import Imputer
    # 缺失值处理
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)

    # 第一步，找出负样本！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    field_pu = [
        '分摊金额'
        , '流量包总收入'
        , 'ONEHOT融合标识_单C'
        , '手机注册时长_天'
        , '上网流量费元_LAST2'
    ]
    df_train_pu = df_train_data_org[field_pu]  # P
    df_test_pu = df_test_data_org[field_pu]  # u

    pu_P_index = np.where(df_train['LABEL'] == 1)[0]  # len(pu_P_index)=5645
    pu_S_size = np.ceil(len(pu_P_index) * 0.15)  # 847
    np.random.seed(42)
    np.random.shuffle(pu_P_index)

    # 随机提取 15%的正样本
    pu_S_index = pu_P_index[:pu_S_size]  # len(pu_S_index)=847

    # 重新生成PS 和 US
    pu_PS_index = list(set(pu_P_index) - set(pu_S_index))  # len(pu_PS_index)=4798
    pu_US_index = list(set(np.arange(len(df_train_pu))) - set(pu_PS_index))  # len(pu_US_index)=1373894

    # pu标签 不同于 train的标签
    pu_label = df_train['LABEL'].copy()
    pu_label = pu_label.reset_index(drop=True)
    pu_label[pu_PS_index] = 1
    pu_label[pu_US_index] = 0
    """ 将原始训练集，更根据S， 重新设置正负样本"""

    from sklearn.naive_bayes import GaussianNB

    # 没有用pu算法，使用原始数据集，训练一个模型
    NB_clf = GaussianNB().fit(df_train_pu, df_train.LABEL)
    out_NB = NB_clf.predict_proba(df_test_pu)[:, 1]
    print('auc:', roc_auc_score(df_test.LABEL, out_NB))

    # 生成一个新的对照预测集合 含id,label,和正常预测的prob
    df_test_prob = pd.concat([df_test[['LABEL', 'SERV_ID']].reset_index(drop=True), pd.DataFrame({'prob': out_NB})],
                             axis=1)
    df_test_prob = df_test_prob.sort_values('prob', ascending=False)

    print('测试集正样本占比:'
          , len(df_test_prob[df_test_prob['LABEL'] == 1])
          , len(df_test_prob)
          , len(df_test_prob[df_test_prob['LABEL'] == 1]) / len(df_test_prob)
          )

    # pu算法，用朴素贝叶斯建模， 原始训练集 使用pu_label,训练新的模型
    NB_clf = GaussianNB().fit(df_train_pu, pu_label)
    out_NB = NB_clf.predict_proba(df_test_pu)[:, 1]
    print('auc:', roc_auc_score(df_test.LABEL, out_NB))

    # 仅使用us集合进行预测
    out_NB = NB_clf.predict_proba(df_train_pu.iloc[pu_US_index])[:, 1]
    th = 0.00009
    # th = 0.000001
    out_1_index = np.where(out_NB >= th)[0]
    out_0_index = np.where(out_NB < th)[0]
    print(len(out_1_index), " are >=", th)
    print(len(out_0_index), " are <", th)

    # 交集
    jiaoji_list = list(set(out_0_index).intersection(set(pu_S_index)))
    print(
        'S中落到负样本数=', len(jiaoji_list)
        , ',S总长度=', len(pu_S_index)
        , ',比例=', len(jiaoji_list) / len(pu_S_index))

    # 输出：可靠的负样本集合 (Reliable Negative Examples，简称 RN)
    out_RU_index = list(set(out_0_index) - set(jiaoji_list))
    print('输出可靠的负样本集合:RU size=', len(out_RU_index))

    # 输出训练的正负样本
    out_train_index = out_RU_index.copy()
    out_train_index.extend(list(pu_P_index))
    print('输出训练集合,size=', len(out_train_index))
    print(len(pu_P_index), " are 1")
    print(len(out_RU_index), " are 0")

    # 第二步，利用上面找出的负样本建模！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    fields = df_train_data_org.columns
    df_train_data = df_train_data_org[fields]
    df_test_data = df_test_data_org[fields]

    # 用第一步输出的训练集!!!!!!!
    df_train_data = df_train_data.iloc[out_train_index]
    df_train_label = df_train.LABEL.iloc[out_train_index]

    num_train, num_feature = df_train_data.shape

    lgb_train = lgb.Dataset(df_train_data, df_train_label, free_raw_data=False
                            # ,feature_name=list(fields),categorical_feature=[2] #商品ID 是 2
                            )
    lgb_eval = lgb.Dataset(df_test_data, df_test.LABEL, free_raw_data=False, reference=lgb_train
                           # ,feature_name=list(fields),categorical_feature=[2]
                           )

    num_boost_round = 5000
    early_stopping_rounds = 500

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 8,
        'lambda_l2': 20,
        # 'max_bin':200,
        # 'max_depth':7,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1,
        # 'scale_pos_weight':100
    }

    fit_begintime = datetime.datetime.now()
    print('fit开始', fit_begintime)
    # feature_name and categorical_feature
    evals_result = {}
    lgb_model = lgb.train(params,
                          lgb_train,
                          num_boost_round=num_boost_round,
                          valid_sets=[lgb_train, lgb_eval],  # eval training data
                          early_stopping_rounds=early_stopping_rounds,
                          # feature_name=feature_name,#list(fields),#,
                          # categorical_feature=[2],
                          evals_result=evals_result,
                          )
    fit_endtime = datetime.datetime.now()
    print('fit结束', fit_endtime)
    print('耗时(秒):', str((fit_endtime - fit_begintime).seconds))

    print('df_train', df_train.shape)
    print('df_test', df_test.shape)
    print('df_train_data', df_train_data.shape)
    print('df_test_data', df_test_data.shape)

    # 预测评估
    y_pred = lgb_model.predict(df_test_data, num_iteration=lgb_model.best_iteration)
    print('lgb_model.best_iteration', lgb_model.best_iteration)
    print('auc:', roc_auc_score(df_test.LABEL, y_pred))

    df_test_prob = pd.concat([df_test[['LABEL', 'SERV_ID']].reset_index(drop=True), pd.DataFrame({'prob': y_pred})],
                             axis=1)
    df_test_prob = df_test_prob.sort_values('prob', ascending=False)
    print('测试集正样本占比:'
          , len(df_test_prob[df_test_prob['LABEL'] == 1])
          , len(df_test_prob)
          , len(df_test_prob[df_test_prob['LABEL'] == 1]) / len(df_test_prob)
          )

    for top_num in [200, 500, 1000, 2000, 5000, 10000]:
        df_top = df_test_prob.head(top_num)
        print(
            '测试集'
            , 'top:', top_num
            , '成功:', len(df_top[df_top['LABEL'] == 1])
            , '准确率:', len(df_top[df_top['LABEL'] == 1]) / top_num
            , '召回率:', len(df_top[df_top['LABEL'] == 1]) / len(df_test_prob[df_test_prob['LABEL'] == 1])
        )

    # 性能可视化
    print('Plot metrics during training...')
    ax = lgb.plot_metric(evals_result, metric='auc')
    plt.show()

    # 特征重要性
    feature_score = lgb_model.feature_importance()
    df_feature_s = pd.DataFrame({'score': feature_score}, index=fields)
    df_feature_s = df_feature_s.sort_values('score', ascending=False)
    s = ''
    for index in df_feature_s.index:
        s = s + index + ':' + str(df_feature_s.loc[index, 'score']) + ','
    print(s)

    # 特征可视化
    ax = lgb.plot_importance(lgb_model, max_num_features=40)
    plt.show()


def tunning(train):
    y = train["ls_flag"]
    X = train.drop(["ls_flag", "Prd_Inst_Id_a"], axis=1)

    cv_params = {
        'n_estimators': range(160, 170),
        'max_depth': range(6, 12),
        'min_child_weight': range(3, 9),
        'gamma': [0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
        'subsample': [0.5, 0.55, 0.6, 0.65],
        'colsample_bytree': [0.8, 0.9, 0.95],
        'reg_alpha': [1, 2, 3, 4, 5],
        'reg_lambda': [2, 3, 4, 5, 6],
        'learning_rate': [0.01, 0.02, .03, .001, 0.005]
    }

    base_para = {
        "n_estimators": 200,
        'learning_rate': .1,
        'max_depth': 3,
        'seed': 2018,
        'metrics': 'auc'
    }
    for k in cv_params:
        cvp = {k: cv_params[k]}
        print(base_para)
        model = xgb.XGBClassifier(**base_para)
        base_para[k] = xgb_cv(model, X, y, cvp)[k]


def xgb_cv(model, X_train, Y_train, cv_params):
    optimized_GBM = GridSearchCV(
        estimator=model,
        param_grid=cv_params,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=4)
    optimized_GBM.fit(X_train, Y_train)
    evalute_result = optimized_GBM.cv_results_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    return optimized_GBM.best_params_


def Model_Xgb_Metrics(base_para, train_x, train_y, cv_params, isCv=True, cv_folds=5,
                      early_stopping_rounds=50):
    # XGB模型调邮
    model_base = XGBClassifier(**base_para)

    optimized_GBM = GridSearchCV(estimator=model_base, param_grid=cv_params, scoring='roc_auc', cv=5,
                                 verbose=1, n_jobs=4)
    optimized_GBM.fit(train_x, train_y)
    # evalute_result = optimized_GBM.cv_results_
    evalute_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

    return optimized_GBM.best_params_

    # if isCv:
    #     xgb_param = clf.get_xgb_params()
    #     xgtrain = xgb.DMatrix(train_x, label=train_y)
    #     cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds,
    #                       metrics='auc', early_stopping_rounds=early_stopping_rounds)  # 是否显示目前几颗树额
    #     clf.set_params(n_estimators=cvresult.shape[0])
    #
    # clf.fit(train_x, train_y, eval_metric='auc')
    #
    # # 预测
    # train_predictions = clf.predict(train_x)
    # train_predprob = clf.predict_proba(train_x)[:, 1]  # 1的概率
    #
    # # 打印
    # print("\nModel Report")
    # print("Accuracy : %.4g" % metrics.accuracy_score(train_y, train_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, train_predprob))
    #
    # feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature importance')
    # plt.ylabel('Feature Importance Score')
    # plt.show()


def main():
    train, test = load_data_source(trainf="cdma_train0611.csv", testf="cdma_predict0611.csv")
#    print(train.shape,  test.shape)

    # res = mini_sample(train[[x for x in train.columns if x not in [id, flag]]].values,
    #             train[flag].values,
    #             test[[x for x in test.columns if x not in [id, flag]]].values)  # 589
    #    print(train.head())
    # print(len(np.where(res>.5)[0]))
    model_name = single_model_xgb(train, show_learning=False)
    # model_name = single_model_lgb(train)
    #
    # test = load_data("data/chen_hskd_04_qf_531.txt")
    # single_RandomForest(train, test)
    # tunning(train)
    # 587
    predict_our_file = predict_df(test, model=model_name,
                                  test_type=0, out_flag=1)


if __name__ == '__main__':
    stime = time.time()
    print(stime)
    main()
    print("耗时:{}".format(time.time() - stime))

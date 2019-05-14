import pandas as pd


def feature_value_counts(data, limit_counts=2000, only_str=True):
    """
    将字符串类型类别特征替换为按数量多少排序的整数类型
    """
    one_dict = []
    overlimit_dict = []
    if only_str:
        process_columns = data.dtypes[data.dtypes == pd.np.object].index
    else:
        process_columns = data.columns
    for col in process_columns:
        v = data[col].value_counts()
        if v.shape[0] == 1:
            one_dict.append(col)
        elif v.shape[0] >= limit_counts:
            overlimit_dict.append(col)
    return one_dict, overlimit_dict


def transform_line_rate(data, most_freq=None):
    """
    将速率统一转换为M为单位的浮点型
    """
    import re
    reg = r'^\D*(\d+\.?\d*)(\D).*$'
    if not isinstance(data, pd.DataFrame):
        raise Exception('data必须为DataFrame')
    if not most_freq or not re.match(reg, most_freq):
        for i in list(data.LINE_RATE.value_counts().index):
            if re.match(reg, i):
                most_freq = i
                break

    def x(e):
        match = re.match(reg, e)
        if not match:
            match = re.match(reg, most_freq)
        v, t = match.groups()
        t = t.upper()
        v = float(v)
        if t == 'K':
            v = v / 1024
        elif t == 'M':
            pass
        elif t == 'G':
            v = v * 1024
        else:
            raise Exception('{} 不在 K,M,G内'.format(e))
        return v

    data.LINE_RATE = data.LINE_RATE.apply(x)
    return data


class DanBroadBand(object):
    """
    针对单宽转融的数据处理类
    """

    def __init__(self, data):
        if isinstance(data, str):
            data = pd.read_csv(data, sep='|')
        if isinstance(data, pd.DataFrame):
            data.rename(lambda x: x.upper(), axis=1, inplace=True)
            data.set_index('PRD_INST_ID', inplace=True)
            transform_line_rate(data)
            value_count_dict = {col: data[col].value_counts() for col in data.columns}
            self.value_count_dict = {k: list(v.index) for k, v in value_count_dict.items() if v.shape[0] > 1}
            self.drop_columns = [k for k, v in value_count_dict.items() if len(v) == 1]
            self.bool_columns = [i for i in data.columns if i.endswith("_FLAG") and i not in self.drop_columns]
            self.process_str_columns = [i for i in data.dtypes[data.dtypes == pd.np.object].index if
                                        i not in (self.bool_columns + self.drop_columns)]
        else:
            raise Exception("data必須為文件路徑或者DataFrame")

    def replace_bool(self, data, flag_true='T'):
        for i in self.bool_columns:
            data[i] = data[i].apply(lambda x: x == flag_true)
        return data

    def process_value_counts(self, data):
        for col in self.process_str_columns:
            kl = self.value_count_dict.get(col)
            data[col] = data[col].fillna(kl[0])
            def x(e):
                if e not in kl:
                    kl.append(e)
                return kl.index(e)
            data[col] = data[col].apply(x)
        return data.fillna(0)

    def transform(self, data):
        if isinstance(data, str):
            data = pd.read_csv(data, sep='|')
        if isinstance(data, pd.DataFrame):
            data.rename(lambda x: x.upper(), axis=1, inplace=True)
            data.set_index('PRD_INST_ID', inplace=True)
            data = transform_line_rate(data)
            return self.process_value_counts(
                self.replace_bool(
                    data.drop(
                        columns=self.drop_columns
                    )
                )
            )
        else:
            raise Exception('data必須為文件路徑或者DataFrame')


# pca = PCA()
# scaler = MinMaxScaler()
# def load_data(fname='dan_bd_train.csv'):
#     data = pd.read_csv(fname).drop(columns=['CUST_TYPE_ID', 'REMOVE_TYPE_ID', 'PRD_INST_FLAG', 'PRD_INST_NBR', 'OVER_PER_HOUR_AMT'])
#     X = data.drop(columns=['LABEL', 'PRD_INST_ID'])
#     y = data.LABEL
#     return data, X, y
#
# data, X, y = load_data()
# PCA()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# params = {
#     'learning_rate': 0.1,
#      'n_estimators': 87,
#      'max_depth': 9,
#      'min_child_weight': 1,
#      'gamma': 0,
#      'subsample': 0.8,
#      'colsample_bytree': 0.9,
#      'scale_pos_weight': 3,
#      'n_jobs': 32,
#      'objective': 'binary:logistic',
#      'reg_alpha': 0,
#      'reg_lambda': 0.005
# }
# dv = DictVectorizer(sparse=False)
# n = data.to_dict(orient='records')
# dv.fit(n)
# dvFun = FunctionTransformer(func=lambda x: dv.transform(x.to_dict(orient='records')), validate=False)
# pline = pipeline.Pipeline(steps=[
#     ("dv", dvFun),
#     ('nanprocess', Imputer(strategy="most_frequent", axis=0)),
#     ('xgb', XGBClassifier(**params))
# ])
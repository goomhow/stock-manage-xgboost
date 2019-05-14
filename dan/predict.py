from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.externals import joblib

ID_COLUMN = 'PRD_INST_ID'


def generate_filename(fmt):
    date = datetime.now()
    date_str = date.strftime('%m%d%H%M')
    return fmt.format(date_str)


def predict_df(bst, data, rm_col=None):
    label = 'LABEL'
    if isinstance(data, str):
        print('*'*10+'LOAD DATA'+'*'*10)
        data = pd.read_csv(data, index_col=ID_COLUMN)
    if rm_col:
        data.drop(columns=rm_col, inplace=True)
    id = pd.read_csv('../prd_inst_id.csv', index_col=ID_COLUMN)
    data = data.loc[id.index, :].dropna()
    print('*' * 10 + 'LOAD MODEL' + '*' * 10)
    if isinstance(bst, str):
        bst = joblib.load(bst)
    X_test = data.drop(columns=label)
    y_pred = bst.predict(X_test)
    y_test_pred1 = bst.predict_proba(X_test)
    out = pd.DataFrame(
        {
            'LATN_ID': data["LATN_ID"].astype(np.int64),
            'PREDICT': y_pred.astype(np.int32),
            'POSSIBILITY': y_test_pred1[:, 1].astype(np.float)
        }
    )
    out = out[out['PREDICT'] == 1].sort_values(by=['PREDICT', 'POSSIBILITY'], ascending=False)
    out.to_csv(generate_filename("result_{}.csv"), header=True, index=True,index_label=ID_COLUMN)
    return out
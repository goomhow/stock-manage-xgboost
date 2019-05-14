from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.externals import joblib


def generate_filename(fmt):
    date = datetime.now()
    date_str = date.strftime('%m%d%H%M')
    return fmt.format(date_str)


def predict_df(model, df_test='', label='LABEL', pfname="bd_predict.csv"):
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
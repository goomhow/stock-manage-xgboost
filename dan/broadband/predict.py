from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.externals import joblib


def get_transformed_data(fname='dan_bd_train.csv'):
    rm_cols = ['CUST_TYPE_ID', 'REMOVE_TYPE_ID', 'PRD_INST_FLAG', 'PRD_INST_NBR', 'OVER_PER_HOUR_AMT']
    data = pd.read_csv(fname).drop(columns=rm_cols)
    bool_col = [col for col in data.columns if col[-5:] == '_FLAG']
    for col in bool_col:
        data[col] = data[col].apply(lambda x: x == 'T')
    X = data.drop(columns=['LABEL', 'PRD_INST_ID'])
    y = data['LABEL']
    return data, X, y

def generate_filename(fmt):
    date = datetime.now()
    date_str = date.strftime('%m%d%H%M')
    return fmt.format(date_str)


def predict_df(model, pfname="dan_bd_predict.csv"):
    bst = joblib.load(model)
    df_test,X_test,__y = get_transformed_data(fname=pfname)
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
import pandas as pd
import os
import time

ID_COLUMN = 'PRD_INST_ID'
LABEL = 'LABEL'


def running_time(func):
    def wrapper(*param, **kwargs):
        startTime = time.time()
        x = func(*param, **kwargs)
        endTime = time.time()
        msecs = int(endTime - startTime)/60
        print("time is %d min" %msecs)
        return x
    return wrapper


@running_time
def transformed_data(month):
    dpath = 'data/dan_{}.csv'.format(month)
    if not os.path.exists(dpath):
        fname = 'train_{}.csv'.format(month)
        lname = 'label/label_{}.csv'.format(month)
        data = pd.read_csv(fname, index_col=ID_COLUMN)
        data.drop(columns='LABEL', inplace=True)
        label = pd.read_csv(lname, index_col=ID_COLUMN)
        df = label.join(data).dropna()
        del data
        del label
        df.to_csv(dpath, index=True, index_label=ID_COLUMN)


if __name__ == '__main__':
    for i in range(201805, 201808):
        transformed_data(i)

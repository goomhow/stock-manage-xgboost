# -*- coding: utf-8 -*-
'''政企用户挖掘'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import os
from sqlalchemy import types, create_engine
import random as rd
import numpy as np
from imblearn.over_sampling import SMOTE
import cx_Oracle
#--------------------------模型参数区域---------------------------------
engine = create_engine("oracle://anti_fraud:at_87654@133.0.176.69:1521/htxxsvc2")
file_path = 'F:/DSqiancaiyang.csv'
chunk_size = 200000
data_cnt = chunk_size*10
DB_path = 'TM_UD_MODEL_201710_RESULT'

#--------------------------功能区域---------------------------------
def chunk_read_data(file_path,chunk_size,data_cnt):
    '''文件批量读取'''
    data = pd.read_csv(file_path,header=0,iterator=True,chunksize=chunk_size)
    train = data.get_chunk(data_cnt)
    train = train.fillna(-1)
    print('文件读取完毕,总计{}条'.format(train.shape[0]))
    return train

data = chunk_read_data(file_path,chunk_size,data_cnt)

Prd_Inst_Id = data['BILLING_NBR']
Gs_id = data['CORP']
train_y = data['LABEL']
train_x = data.drop(['BILLING_NBR','LABEL','CORP'],axis = 1)

x_train,x_test,y_train,y_test = train_test_split(train_x,train_y,test_size = 0.2,random_state = 5)

model = xgb.XGBClassifier(learning_rate =0.01,
                  n_estimators=5000,
                  max_depth=5,
                  min_child_weight=3,
                  gamma=0.3,
                  subsample=0.85,
                  colsample_bytree=0.75,
                  objective= 'binary:logistic',
                  scale_pos_weight=1,
                  seed=27,
                  nthread=12,
                  reg_alpha=0.0005)
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
train_report = metrics.classification_report(y_train,y_train_pred)
test_report = metrics.classification_report(y_test,y_test_pred)
print(train_report)
print(test_report)

data = chunk_read_data('f:/ds_all.csv',chunk_size,data_cnt)
Prd_Inst_Id = data['BILLING_NBR']
Gs_id = data['CORP']
train_y = data['LABEL']
train_x = data.drop(['BILLING_NBR','LABEL','CORP'],axis = 1)
y_train_pred = model.predict(train_x)

train_report = metrics.classification_report(train_y,y_train_pred)
data['pred_label'] = y_train_pred
out_data = pd.concat([data,y_train_pred],axis=1)
data.to_csv('F:/jieguo.csv')
print(train_report)
print(test_report)
# def output_into_DB(engine,data_day,DB_path):
#     dtyp = {c: types.VARCHAR(data_day[c].str.len().max())
#             for c in data_day.columns[data_day.dtypes == 'object'].tolist()}
#     print(dtyp)
#     data_day.to_sql(DB_path, engine, index=False, if_exists='append',dtype=dtyp)
#     # data_day.to_csv(os.path.join('/data1/fxmx_{}.csv'.format(now)), index=False)
#     print('预测结果写入数据库完毕')


# def model_train_type(data,type):
#     '''type为数据形式,0全量，1本网，2异网，3固话，4非本网,5固话＋本网'''
#     if type == 0:
#         train = data
#         print('全量号码模式')
#     elif type == 1:
#         train = data.loc[data['IS_OTHER_NET'] == 0]
#         print('本网号码模式')
#     elif type == 2:
#         train = data.loc[data['IS_OTHER_NET'] == 1]
#         print('异网号码模式')
#     elif type == 3:
#         train = data.loc[data['IS_OTHER_NET'] == 2]
#         print('固话号码模式')
#     elif type == 4:
#         train = data.loc[data['IS_OTHER_NET'] >=1]
#         print('非电信号码模式')
#     elif type == 5:
#         train = data.loc[data['IS_OTHER_NET'] !=1]
#         print('本网移动+固话模式')
#     else:
#         print('请重新输入文件读取模式类型')
#     print('共计{}条数据'.format(train.shape[0]))
#     return train

def train_model(data,model_type,gs_name=None):
    '''1：全量数据模式 2：单公司训练模式'''
    if model_type == 1:
        new_data = data
        Prd_Inst_Id = new_data['BILLING_NBR']
        gs_id = new_data['CORP']
        train_y = new_data['LABEL']
        train_x = new_data.drop(['BILLING_NBR', 'LABEL', 'CORP'], axis=1)

    elif model_type == 2:
        new_data = data
        Prd_Inst_Id = new_data['BILLING_NBR']
        gs_id = new_data['CORP']
        new_data = new_data.loc[data['CORP'] == gs_name]
        train_y = new_data['LABEL']
        train_x = new_data.drop(['BILLING_NBR', 'LABEL', 'CORP'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=5)
    model = xgb.XGBClassifier(learning_rate=0.01,
                              n_estimators=5000,
                              max_depth=5,
                              min_child_weight=3,
                              gamma=0.3,
                              subsample=0.85,
                              colsample_bytree=0.75,
                              objective='binary:logistic',
                              scale_pos_weight=1,
                              seed=27,
                              nthread=12,
                              reg_alpha=0.0005)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_report = metrics.classification_report(y_train, y_train_pred)
    test_report = metrics.classification_report(y_test, y_test_pred)
    print(train_report)
    print(test_report)
    return Prd_Inst_Id,gs_id,train_x,train_y,x_train, x_test, y_train, y_test,model


# if __name__ == '__main__':
    # data = read_data(read_data_path) #读取全量数据
    #
    # '''type为数据形式,0全量，1本网，2异网，3固话，4非本网,5固话＋本网'''
    # train = model_train_type(data,type = 1)
    # Prd_Inst_Id,gs_id,train_x,train_y,x_train, x_test, y_train, y_test,model = train_model(train,1)

data = pd.read_table(read_data_path,sep=',',header=0)
data = data.fillna(-1)
# data = data.loc[data['IS_OTHER_NET'] > 0]
data.head(5)
data.to_csv('F:/test.csv')
'''方式一：全量用户训练'''
Prd_Inst_Id = data['BILLING_NBR']
Gs_id = data['CORP']
train_y = data['LABEL']
train_x = data.drop(['BILLING_NBR','LABEL','CORP'],axis = 1)

x_train,x_test,y_train,y_test = train_test_split(train_x,train_y,test_size = 0.2,random_state = 5)

model = xgb.XGBClassifier(learning_rate =0.01,
                  n_estimators=5000,
                  max_depth=5,
                  min_child_weight=3,
                  gamma=0.3,
                  subsample=0.85,
                  colsample_bytree=0.75,
                  objective= 'binary:logistic',
                  scale_pos_weight=1,
                  seed=27,
                  nthread=12,
                  reg_alpha=0.0005)
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
train_report = metrics.classification_report(y_train,y_train_pred)
test_report = metrics.classification_report(y_test,y_test_pred)
print(train_report)
print(test_report)

'''全量用户结果输出'''
train_x_pred = model.predict(train_x)
train_x_pred = pd.DataFrame(train_x_pred,columns = ['flag'])
train_x_pred = train_x_pred.set_index(np.arange(1,len(train_x_pred)+1,1))
data = data.set_index(np.arange(1,len(data)+1,1))
out_data = pd.concat([data,train_x_pred],axis=1)
# flag_all_data = out_data[out_data['flag'] == 1]
print(out_data.shape)
out_data.to_csv('F:/all_predict_12.csv')

'''方式二：电信用户提取'''
gs_name = 'DS'
data = data.loc[data['CORP'] == gs_name]
# new_data = data.loc[data['IS_OTHER_NET'] == 0]
train_x = data.drop(['BILLING_NBR','LABEL','CORP'],axis = 1)
train_y = data['LABEL']

x_train,x_test,y_train,y_test = train_test_split(train_x,train_y,test_size = 0.2,random_state = 5)
sm = SMOTE(random_state=42)
x_train, y_train = sm.fit_sample(x_train, y_train)
model = xgb.XGBClassifier(learning_rate =0.01,
                  n_estimators=5000,
                  max_depth=5,
                  min_child_weight=3,
                  gamma=0.3,
                  subsample=0.85,
                  colsample_bytree=0.75,
                  objective= 'binary:logistic',
                  scale_pos_weight=1,
                  seed=27,
                  nthread=12,
                  reg_alpha=0.0005)
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
train_report = metrics.classification_report(y_train,y_train_pred)
test_report = metrics.classification_report(y_test,y_test_pred)
print(train_report)
print(test_report)

'''电信用户结果输出'''
train_x_pred = model.predict(train_x)
train_report = metrics.classification_report(train_y,train_x_pred)
print(train_report)
train_x_pred = pd.DataFrame(train_x_pred,columns = ['flag'])
train_x_pred = train_x_pred.set_index(np.arange(1,len(train_x_pred)+1,1))
out_data = pd.concat([data,train_x_pred],axis=1)
# flag_all_data = out_data[out_data['flag'] == 1]
print(out_data.shape)
out_data.to_csv('F:/dx_predict_12.csv')

"""电信用户套用YZ"""
data_input = 'F:/all201801.csv'   #test_vol_nbr_inf.txt   vol_nbr_inf.txt
data = pd.read_table(data_input,sep=',',header=0)
data = data.fillna(-1)

gs_name = 'YZ'
data = data.loc[data['CORP'] == gs_name]
# data = data.loc[data['IS_OTHER_NET'] > 0]
data = data.set_index(np.arange(1,len(data)+1,1))
train_x = data.drop(['BILLING_NBR','LABEL','CORP'],axis = 1)
train_y = data['LABEL']

'''邮政用户结果输出'''
train_x_pred = model.predict(train_x)
train_report = metrics.classification_report(train_y,train_x_pred)
print(train_report)

train_x_pred = pd.DataFrame(train_x_pred,columns = ['flag']).set_index(np.arange(1,len(train_x_pred)+1,1))
out_data = pd.concat([data,train_x_pred],axis=1)
# flag_all_data = out_data[out_data['flag'] == 1]
print(out_data.shape)
out_data.to_csv('F:/yz_predict.csv')

"""电信用户套用DS"""
data_input = 'F:/all201801.csv'   #test_vol_nbr_inf.txt   vol_nbr_inf.txt
data = pd.read_table(data_input,sep=',',header=0)
data = data.loc[data['IS_OTHER_NET'] > 0]
data = data.fillna(-1)
data.head(5)
gs_name = 'DS'
data = data.loc[data['CORP'] == gs_name]
data = data.set_index(np.arange(1,len(data)+1,1))
train_x = data.drop(['BILLING_NBR','LABEL','CORP'],axis = 1)
train_y = data['LABEL']

'''邮政用户结果输出'''
train_x_pred = model.predict(train_x)
train_report = metrics.classification_report(train_y,train_x_pred)
print(train_report)

train_x_pred = pd.DataFrame(train_x_pred,columns = ['flag']).set_index(np.arange(1,len(train_x_pred)+1,1))
out_data = pd.concat([data,train_x_pred],axis=1)
# flag_all_data = out_data[out_data['flag'] == 1]
print(out_data.shape)
out_data.to_csv('F:/ds_predict.csv')

"""电信用户套用AJ"""
data_input = 'F:/all201801.csv'   #test_vol_nbr_inf.txt   vol_nbr_inf.txt
data = pd.read_table(data_input,sep=',',header=0)
data = data.loc[data['IS_OTHER_NET'] > 0]
data = data.fillna(-1)
data.head(5)
gs_name = 'AJ'
data = data.loc[data['CORP'] == gs_name]
data = data.set_index(np.arange(1,len(data)+1,1))
train_x = data.drop(['BILLING_NBR','LABEL','CORP'],axis = 1)
train_y = data['LABEL']

'''邮政用户结果输出'''
train_x_pred = model.predict(train_x)
train_report = metrics.classification_report(train_y,train_x_pred)
print(train_report)

train_x_pred = pd.DataFrame(train_x_pred,columns = ['flag']).set_index(np.arange(1,len(train_x_pred)+1,1))
out_data = pd.concat([data,train_x_pred],axis=1)
# flag_all_data = out_data[out_data['flag'] == 1]
print(out_data.shape)
out_data.to_csv('F:/aj_predict_yw.csv')


"""电信用户套用zJ"""
data_input = 'F:/all201801.csv'   #test_vol_nbr_inf.txt   vol_nbr_inf.txt
data = pd.read_table(data_input,sep=',',header=0)
data = data.fillna(-1)
data = data.loc[data['IS_OTHER_NET'] == 0]
data.head(5)
gs_name = 'ZJ'
data = data.loc[data['CORP'] == gs_name]
# data = data.set_index(np.arange(1,len(data)+1,1))
train_x = data.drop(['BILLING_NBR','LABEL','CORP'],axis = 1)
train_y = data['LABEL']

'''邮政用户结果输出'''
train_x_pred = model.predict(train_x)
train_report = metrics.classification_report(train_y,train_x_pred)
print(train_report)

train_x_pred = pd.DataFrame(train_x_pred,columns = ['flag']).set_index(np.arange(1,len(train_x_pred)+1,1))
out_data = pd.concat([data,train_x_pred],axis=1)
# flag_all_data = out_data[out_data['flag'] == 1]
print(out_data.shape)
out_data.to_csv('F:/Zj_predict.csv')
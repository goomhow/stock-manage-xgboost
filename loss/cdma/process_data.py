import pandas as pd
import os
import time
LABEL = 'LABEL'
ID_COLUMN = 'PRD_INST_ID'

rm_col = ['BIL_1X_DUR_A',
            'BIL_1X_DUR_B',
            'BIL_1X_DUR_C',
            'CENTREX_FLAG',
            'CHARGE_FT_BEFORE_B',
            'EXACT_FLUX_A',
            'EXACT_FLUX_B',
            'EXACT_FLUX_C',
            'EXTREM_PACK_FLUX_A',
            'EXTREM_PACK_FLUX_B',
            'EXTREM_PACK_FLUX_C',
            'F1X_FLUX_A',
            'F1X_FLUX_B',
            'F1X_FLUX_C',
            'F3G_FLUX_A',
            'F3G_FLUX_B',
            'FIN_OWE_AMT_B',
            'FIN_OWE_AMT_C',
            'HDAY_DAYS_A',
            'HDAY_DUR_A',
            'HDAY_DUR_C',
            'HDAY_FLUX_A',
            'HDAY_FLUX_B',
            'HOME_FLUX_A',
            'IN_EXACT_FLUX_A',
            'IN_EXACT_FLUX_B',
            'IN_PACK_EXACT_FLUX_A',
            'IN_PACK_EXACT_FLUX_B',
            'OFFICE_DUR_A',
            'OFFICE_FLUX_A',
            'OFFICE_FLUX_B',
            'OFF_DUR_A',
            'OFF_FLUX_A',
            'OFF_FLUX_B',
            'OFF_FLUX_C',
            'ON_DUR_A',
            'ON_DUR_B',
            'ON_FLUX_A',
            'ON_FLUX_B',
            'OWE_AMT_A',
            'OWE_AMT_C',
            'OWE_DUR_A',
            'OWE_DUR_B',
            'O_INET_PP_SMS_CNT_A',
            'O_INET_PP_SMS_CNT_B',
            'O_ONET_PP_SMS_CNT_A',
            'O_ONET_PP_SMS_CNT_B',
            'O_ONET_PP_SMS_CNT_C',
            'O_TOL_DSTN_A',
            'O_TOL_DUR_A',
            'O_TOL_DUR_B',
            'PACK_CNT_A',
            'PACK_CNT_B',
            'PACK_CNT_C',
            'PACK_FLAG_A',
            'PACK_FLAG_B',
            'PACK_FLAG_C',
            'PP_SMS_AMT_A',
            'PP_SMS_AMT_B',
            'SP_SMS_AMT_A',
            'SP_SMS_AMT_C',
            'TDD_BIL_DUR_A',
            'TDD_FLUX_A',
            'TDD_FLUX_B',
            'TDD_FLUX_C',
            'TOTAL_1X_CNT_A',
            'TOTAL_1X_CNT_B',
            'TOTAL_1X_CNT_C',
            'TOTAL_3G_CNT_A',
            'TOTAL_3G_CNT_B',
            'TOTAL_FLUX_A',
            'TOTAL_FLUX_B',
            'TOTAL_TDD_CNT_A',
            'T_CALL_DSTN_A',
            'T_ONET_PP_SMS_CNT_A',
            'T_ONET_PP_SMS_CNT_B',
            'T_SP_SMS_CNT_A',
            'T_SP_SMS_CNT_B',
            'VIP_FLAG',
            'WDAY_DUR_A',
            'WDAY_FLUX_A']


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
    fname = 'cdma_{}.csv'.format(month)
    xName = 'cdma_{}_x.csv'.format(month)
    lName = 'label/label_{}.csv'.format(month)
    print("load {} data".format(fname))
    data = pd.read_csv(fname, index_col='PRD_INST_ID')
    if os.path.exists(lName):
        print("load {} lable".format(lName))
        label = pd.read_csv(lName, index_col=ID_COLUMN)
        data[LABEL] = label[label['LABEL'] > -1]
        data.dropna(inplace=True)
    data.sort_index(inplace=True)
    print("start caculate")
    common = [i[:-2] for i in data.columns if i.endswith('_A') and i not in {'STD_PRD_INST_STAT_ID_A', 'PACK_FLAG_A'}]
    NAME_A = [i+'_A' for i in common]
    NAME_B = [i+'_B' for i in common]
    NAME_C = [i+'_C' for i in common]
    A = data[NAME_A].rename(columns=lambda x: x[:-2])
    B = data[NAME_B].rename(columns=lambda x: x[:-2])
    C = data[NAME_C].rename(columns=lambda x: x[:-2])
    B_A = B - A
    C_B = C - B
    data[NAME_A] = B_A.rename(columns=lambda x: x+'_A')
    data[NAME_B] = C_B.rename(columns=lambda x: x+'_B')
    print("start save {}".format(xName))
    data.to_csv(xName, index=True, header=True, index_label='PRD_INST_ID')


if __name__ == '__main__':
    for i in range(201805, 201808):
        transformed_data(i)
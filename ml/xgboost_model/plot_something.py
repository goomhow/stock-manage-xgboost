import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

data = [(0.13508634, 'OFR_ID'),
        (0.07454757, 'BALANCE_04'),
        (0.0510958, 'INNET_DUR'),
        (0.048875146, 'BALANCE_03'),
        (0.04675826, 'FLUX_4'),
        (0.034638055, 'FLUX_3'),
        (0.029117549, 'INV_BILL_AMT_04'),
        (0.02764403, 'AGE'),
        (0.026606342, 'BRD_BND_DUR_4'),
        (0.026045991, 'CHARGE_BEFORE_04'),
        (0.022953678, 'BRD_BND_DUR_3'),
        (0.01994438, 'CHARGE_BEFORE_03'),
        (0.019031214, 'CHARGE_FT_04'),
        (0.018823676, 'INV_BILL_AMT_03'),
        (0.017495435, 'CHARGE_FT_03'),
        (0.015793625, 'CHARGE_2809_04'),
        (0.014527644, 'ZC_ZB_04'),
        (0.013552217, 'LINE_RATE'),
        (0.01290885, 'CDMA_03'),
        (0.012390005, 'ZC_ZB_03')]


def bar(titile, data, pic_name='宽带流失预警模型特征重要度', x_name='重要率'):
    plt.barh(range(len(data)), data, height=0.7, color='steelblue', alpha=0.8,tick_label=titile)
    plt.yticks(range(len(titile)),titile)
    xlim = max(data)*1.1
    plt.xlim(0,xlim)
    plt.title(pic_name)
    plt.xlabel(x_name)
    for y,x in enumerate(data):
        plt.text(x+xlim*0.01,y-0.1,str(x))
    plt.show()


def tmp(result, original,features):
    r = pd.read_csv(result)
    o = pd.read_csv(original)
    r.set_index('PRD_INST_ID', inplace=True)
    o.set_index('PRD_INST_ID', inplace=True)
    return r.join(o, how='inner', lsuffix='1')[['LATN_ID', 'POSSIBILITY']+features]


if __name__ == '__main__':
    bar([j for i, j in data], [i for i, j in data])


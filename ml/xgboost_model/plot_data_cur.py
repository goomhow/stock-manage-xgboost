from datetime import datetime
import numpy as np
import pandas as pd
from sklearn import metrics, learning_curve, svm
from sklearn.model_selection import *
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.plotting import plot_importance

def top20feature(fmodel,train):
    model = joblib.load(fmodel)
    data = pd.read_csv(train).drop(["MERGE_PROM_INST_ID",'label',"PRD_INST_ID"], axis=1)
    return [j for i, j in sorted(zip(model.feature_importances_(), data.columns), reverse=True)]

if __name__ == '__main__':
    model = joblib.load('cdma_xgb_061216.pkl')
    plot_importance(model)

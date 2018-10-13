# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:48:35 2018

@author: Junjie
"""

import time as time
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler


random_seed = 755

# Load data
train = pd.read_csv("data/Traffic_flow//train.csv")
test = pd.read_csv("data/Traffic_flow/test.csv")

del train['index']
del test['index']

Y = train['Segment23_(t+1)']
Y.describe()

# Feature Selection
def extract(data, slide=range, max_range=None):
    if callable(slide):
        if max_range is None:
            return data['Segment23_(t+1)'].values, data.iloc[:, slide(0, data.shape[1]-1)].values
        if max_range is not None:
            return data['Segment23_(t+1)'].values, data.iloc[:, slide(0, max_range)].values
    if isinstance(slide ,list): 
        return data['Segment23_(t+1)'].values, data.iloc[:, list].values

# Normalization/Standardization
def row_identity(data):
    return data

def row_standardized(data):
    return StandardScaler().fit_transform(data.T).T

def row_minmax(data):
    return MinMaxScaler().fit_transform(data.T).T

def row_normal(data):
    return Normalizer(norm='l1').fit_transform(data)

def log(data):
    return np.log(data+1)


row_transform = {'Identity':row_identity}
col_scalling = {'Identity':None ,'MinMax': MinMaxScaler}

# Feature Extraction
feature_extract = {'Identity': None }


# Cross Validation
kf  = KFold(n_splits=5, random_state=random_seed)

# Hyperparameter Space
tuned_parameters = {"BRidge": []}
score = 'neg_median_absolute_error' # 

# Models
models = {
    "BRidge": "" 
}

# Matrix
cv = []
holdout = []
y_train0, X_train0 = extract(train)
y_test0, X_test0 = extract(test)
for rt_name, transformer in row_transform.items():
    print("# Transforming row value by %s" % rt_name)
    X_train1 = transformer(X_train0)
    X_test1 = transformer(X_test0)
    for sl_name, scaler in col_scalling.items():
        print("# Scaling column value by %s" % sl_name)
        if sl_name != 'Identity':
            sl = scaler()
            sl.fit(X_train1)
            X_train = sl.transform(X_train1)
            X_test = sl.transform(X_test1)
        else: 
            X_train = X_train1.copy()
            X_test = X_test1.copy()
        for mkey, model in models.items():
            print("# Staring fittimg model of %s" % mkey)
            print()
            
            y_train = y_train0
            y_test = y_test0
            print("# Tuning hyper-parameters for %s" % score)
            print()
            print(X_train)
            print(X_test)
            # continue your sklearn code here...
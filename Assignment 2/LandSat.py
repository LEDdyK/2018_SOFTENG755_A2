# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 17:49:41 2018

@author: Junjie
"""

import time as time
import numpy as np
import pandas as pd

from timeit import default_timer as timer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

random_seed = 755

# Load data
train = pd.read_csv("data/Landsat/train.csv")
test = pd.read_csv("data/Landsat/test.csv")

Y = train.label

del train['index']
del test['index']

# Feature Selection/Extraction
def extract(data, slide=range, max_range=None):
    if callable(slide):
        if max_range is None:
            return data.label.values, data.iloc[:, slide(0, data.shape[1]-1)].values
        if max_range is not None:
            return data.label.values, data.iloc[:, slide(0, max_range)].values
    if isinstance(slide ,list): 
        return data.label.values, data.iloc[:, list].values

# Normalization/Standardization
def row_identity(data):
    return data.T.T
def row_standardized(data):
    return StandardScaler().fit_transform(data.T).T
def row_minmax(data):
    return MinMaxScaler().fit_transform(data.T).T
def row_max(data):
    return data/255.

# data transformation functions
row_transform = {'Identity':row_identity }
col_scaling = {'Identity':None, 'ZScore':StandardScaler}

# Feature Extraction
feature_extract = {'Identity': None}

# Cross Validation
skf = StratifiedKFold(n_splits=5, random_state=random_seed)

# Hyperparameter Space
tuned_parameters = {
        "Logistics" :  [{'max_depth': [1,3,5,7,9,11]}],
        "Neural": [{'alpha': np.logspace(0, -5, num=5)}]
}
score = 'f1_micro' # f1_macro

# Models
models = {
    "Logistics": LogisticRegression(),
    "Neural": MLPClassifier()
}

# Matric
cv = []
holdout = []

# Split datasets into features (X) and outputs (y)
y_train0, X_train0 = extract(train)
y_test0, X_test0 = extract(test)

# transform the data via rows: rt_name = key, transformer = value
for rt_name, transformer in row_transform.items():
    print("# Transforming row value by %s" % rt_name)
    X_train1 = transformer(X_train0)
    X_test1 = transformer(X_test0)
    
    # scale the data via columns: sl_name = key, scaler = value
    for sl_name, scaler in col_scaling.items():
        print("# Scaling column value by %s" % sl_name)
        if sl_name != 'Identity':
            sl = scaler()
            sl.fit(X_train1)
            X_train2 = sl.transform(X_train1)
            X_test2 = sl.transform(X_test1)
        else: 
            X_train2 = X_train1.copy()
            X_test2 = X_test1.copy()
            
        # apply feature selection: fe_name = key, fe = value
        for fe_name, fe in feature_extract.items():
            print("# Features Selection/Extraction value by %s" % fe_name)
            if fe_name in ['PCA1', 'PCA2','PCA3']:
                fe.fit(X_train2)
                X_train = fe.transform(X_train2)
                X_test = fe.transform(X_test2)
            else: 
                X_train = X_train2.copy()
                X_test = X_test2.copy()   
                
            # apply the models to the data: mkey = key, model = value
            for mkey, model in models.items():
                print("# Staring fittimg model of %s" % mkey)
                print()
                y_train = y_train0
                y_test = y_test0
                #y_train, X_train = extract(train)
                print("# Tuning hyper-parameters for %s" % score)
                print()
                print(X_train)
                print(X_test)
                # continue your TF code here...





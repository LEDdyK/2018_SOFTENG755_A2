# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 20:50:23 2018

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



random_seed = 755

# load data
train = pd.read_csv("data/Occupancy_sensor/train.csv")
test = pd.read_csv("data/Occupancy_sensor/test.csv")

del train['index']
del test['index']
del train['date']
del test['date']

# Feature Selection/Extraction
def extract(dat, slide=range, max_range=None):
    data = dat.copy()
    y = data['Occupancy']
    del data['Occupancy']
    if callable(slide):
        if max_range is None:
            return y.values, data.iloc[:, slide(0, data.shape[1])].values
        if max_range is not None:
            return y.values, data.iloc[:, slide(0, max_range)].values
    if isinstance(slide ,list): 
        return y.values, data.iloc[:, list].values


col_scalling = {'Identity':None , 'ZScore':  StandardScaler}


# Cross Validation
skf  = StratifiedKFold(n_splits=5, random_state=random_seed)

# Hyperparameter Space
score = 'f1_micro' 

# Models
models = {
    "kmeans": "kmeans",
    "GMM": "gmm"
}

# train/test 
cv = []
holdout = []
Y = train.Occupancy
y_train0, X_train0 = extract(train)
y_test0, X_test0 = extract(test)
# Hyperparameter Space
tuned_parameters = {
        "kmeans": [],
        "GMM": []
}
for sl_name, scaler in col_scalling.items():
    print("# Scaling column value by %s" % sl_name)
    if sl_name != 'Identity':
        sl = scaler()
        sl.fit(X_train0)
        X_train = sl.transform(X_train0)
        X_test = sl.transform(X_test0)
    else: 
        X_train = X_train0.copy()
        X_test = X_test0.copy()
    for mkey, model in models.items():
        print("# Staring fittimg model of %s" % mkey)
        print()
        
        y_train = y_train0
        y_test = y_test0
        print("# Tuning hyper-parameters for %s" % score)
        print()
        print(X_train)
        print(X_test)









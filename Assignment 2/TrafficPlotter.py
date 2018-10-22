# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 20:10:24 2018

@author: Lite
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the cross validation results
cv_results = pd.read_csv("data/Traffic_flow/cv_result.csv")
# load the best predictors
ho_results = pd.read_csv("data/Traffic_flow/ho_result.csv")

# remove unimportant cells
cv_results = cv_results[cv_results['sl_name']=='MinMax']
ho_results = ho_results[ho_results['sl_name']=='MinMax']
cv_results.reset_index(drop=True,inplace=True)
step = 0
for params in cv_results['params']:
    if "'normalize': True" in params:
        cv_results = cv_results.drop(index=step)
    step += 1
cv_results.reset_index(drop=True,inplace=True)

# sort the items by lambda_1 in params
step = 0
fill_list = [0,0,0,0,0,0,0,0,0,0,0]
x_axis = np.arange(100, 301, 20)
for params in cv_results['params']:
    if '100' in params:
        fill_list[0] = cv_results['mean_validation_score'][step]*-1
    elif '120' in params:
        fill_list[1] = cv_results['mean_validation_score'][step]*-1
    elif '140' in params:
        fill_list[2] = cv_results['mean_validation_score'][step]*-1
    elif '160' in params:
        fill_list[3] = cv_results['mean_validation_score'][step]*-1
    elif '180' in params:
        fill_list[4] = cv_results['mean_validation_score'][step]*-1
    elif '200' in params:
        fill_list[5] = cv_results['mean_validation_score'][step]*-1
    elif '220' in params:
        fill_list[6] = cv_results['mean_validation_score'][step]*-1
    elif '240' in params:
        fill_list[7] = cv_results['mean_validation_score'][step]*-1
    elif '260' in params:
        fill_list[8] = cv_results['mean_validation_score'][step]*-1
    elif '280' in params:
        fill_list[9] = cv_results['mean_validation_score'][step]*-1
    elif '300' in params:
        fill_list[10] = cv_results['mean_validation_score'][step]*-1
    step += 1
    
# plot results
fig, ax = plt.subplots()
plt.plot(x_axis, fill_list)
plt.xticks(x_axis)
plt.xlabel('Lambda 1')
plt.ylabel('mean absolute error')
plt.title('Hyperparameter Results - Bayesisan Ridge Regression')
plt.show
plt.savefig('data/Traffic_flow/plots/BRR_MAS.png')

# plot best model against true results
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

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

random_seed = 755

# Load data
train = pd.read_csv("data/Traffic_flow/train.csv")
test = pd.read_csv("data/Traffic_flow/test.csv")
del train['index']
del test['index']

# Split datasets into features (X) and outputs (Y)
Y_train, X_train = extract(train)
Y_test, X_test = extract(test)

# preprocess data
X_train = row_identity(X_train)
X_test = row_identity(X_test)
sl = MinMaxScaler()
sl.fit(X_train)
X_train = sl.transform(X_train)
X_test = sl.transform(X_test)

# modelling best results according to ho_result
model = BayesianRidge(fit_intercept=False, normalize=False, lambda_1=180)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print("mean absolute error:")
err = mean_absolute_error(Y_test, y_pred)
print(err)
print("r2_score:")
r2s = r2_score(Y_test, y_pred)
print(r2s)

# plotting errors
fig, ax = plt.subplots()
test_err = Y_test - y_pred
plt.scatter(range(1,751), test_err, s=50, alpha=0.3, marker='s')
plt.ylim(min(test_err), max(test_err))
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('index')
plt.ylabel('error')
plt.title('Regression Test Errors')
plt.show
plt.savefig('data/Traffic_flow/plots/BRR_TestErrors.png')
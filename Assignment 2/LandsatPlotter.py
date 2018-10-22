# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 13:30:05 2018

@author: Lite
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fill_list(in_list, params, item):
    if '(40' in params or 'lbfgs' in params:
        in_list[0] = item
    elif '(60' in params or 'saga' in params:
        in_list[1] = item
    elif '(80' in params:
        in_list[2] = item
    return in_list

# load the cross validation results
cv_results = pd.read_csv("data/Landsat/cv_result.csv")
# load the best predictors
ho_results = pd.read_csv("data/Landsat/ho_result.csv")

# remove unimportant cells
cv_results = cv_results[cv_results['sl_name']=='ZScore']
ho_results = ho_results[ho_results['sl_name']=='ZScore']

# split the cross validation results
# first split: method
cv_neural = cv_results[cv_results['method']=='Neural']
cv_logreg = cv_results[cv_results['method']=='Logistics']
# second split: features
cv_neural_fs = cv_neural[cv_neural['features']!='all']
cv_neural_fs.reset_index(inplace=True, drop=True)
cv_neural_na = cv_neural[cv_neural['features']=='all']
cv_neural_na.reset_index(inplace=True, drop=True)
cv_logreg_fs = cv_logreg[cv_logreg['features']!='all']
cv_logreg_fs.reset_index(inplace=True, drop=True)
cv_logreg_na = cv_logreg[cv_logreg['features']=='all']
cv_logreg_na.reset_index(inplace=True, drop=True)

# grouping of neural data by hidden_layer_sizes
cv_neural_fs_0005, cv_neural_fs_001 = [0,0,0],[0,0,0]
cv_neural_fs_005, cv_neural_fs_01 = [0,0,0],[0,0,0]
cv_neural_na_0005, cv_neural_na_001 = [0,0,0],[0,0,0]
cv_neural_na_005, cv_neural_na_01 = [0,0,0],[0,0,0]
# fill appropriate groups
step = 0
for params in cv_neural_fs['params']:
    if '0.0005' in params:
        cv_neural_fs_0005 = fill_list(cv_neural_fs_0005,params,cv_neural_fs['mean_validation_score'][step])
    elif '0.001' in params:
        cv_neural_fs_001 = fill_list(cv_neural_fs_001,params,cv_neural_fs['mean_validation_score'][step])
    elif '0.005' in params:
        cv_neural_fs_005 = fill_list(cv_neural_fs_005,params,cv_neural_fs['mean_validation_score'][step])
    elif '0.01' in params:
        cv_neural_fs_01 = fill_list(cv_neural_fs_01,params,cv_neural_fs['mean_validation_score'][step])
    step+=1
step = 0
for params in cv_neural_na['params']:
    if '0.0005' in params:
        cv_neural_na_0005 = fill_list(cv_neural_na_0005, params, cv_neural_na['mean_validation_score'][step])
    elif '0.001' in params:
        cv_neural_na_001 = fill_list(cv_neural_na_001, params, cv_neural_na['mean_validation_score'][step])
    elif '0.005' in params:
        cv_neural_na_005 = fill_list(cv_neural_na_005, params, cv_neural_na['mean_validation_score'][step])
    elif '0.01' in params:
        cv_neural_na_01 = fill_list(cv_neural_na_01, params, cv_neural_na['mean_validation_score'][step])
    step+=1

# create plot
fig, ax = plt.subplots()
index = np.arange(3)
bw=0.2
a=0.7
rects1 = plt.bar(index,cv_neural_fs_0005,bw,color='#63b3c2',label='a=5e-4 w/ fs')
rects2 = plt.bar(index+bw,cv_neural_fs_001,bw,color='#196fef',label='a=1e-3 w/ fs')
rects3 = plt.bar(index+2*bw,cv_neural_fs_005,bw,color='#0e1f60',label='a=5e-3 w/ fs')
rects4 = plt.bar(index+3*bw,cv_neural_fs_01,bw,color='#1df7ff',label='a=1e-2 w/ fs')
rects5 = plt.bar(index+0.25*bw,cv_neural_na_0005,bw/2,alpha=a,color='r',label='w/o fs')
rects6 = plt.bar(index+1.25*bw,cv_neural_na_001,bw/2,alpha=a,color='r')
rects7 = plt.bar(index+2.25*bw,cv_neural_na_005,bw/2,alpha=a,color='r')
rects8 = plt.bar(index+3.25*bw,cv_neural_na_01,bw/2,alpha=a,color='r')
plt.xlabel('hidden layer sizes')
plt.ylabel('mean validation score')
plt.title('Hyperparameter Results - Artificial Neural Network')
plt.xticks(index + 1.5*bw, ('40','60','80'))
plt.ylim(0.893, 0.907)
plt.legend()
plt.tight_layout()
plt.show
plt.savefig('data/Landsat/plots/ANN_hyp_res.png')

# grouping of logistic regression data by solver
cv_logreg_fs_40, cv_logreg_fs_60 = [0,0],[0,0]
cv_logreg_fs_80, cv_logreg_fs_100 = [0,0],[0,0]
cv_logreg_na_40, cv_logreg_na_60 = [0,0],[0,0]
cv_logreg_na_80, cv_logreg_na_100 = [0,0],[0,0]
# fill appropriate groups
step = 0
for params in cv_logreg_fs['params']:
    if '40' in params:
        cv_logreg_fs_40 = fill_list(cv_logreg_fs_40, params, cv_logreg_fs['mean_validation_score'][step])
    elif '60' in params:
        cv_logreg_fs_60 = fill_list(cv_logreg_fs_60, params, cv_logreg_fs['mean_validation_score'][step])
    elif '80' in params:
        cv_logreg_fs_80 = fill_list(cv_logreg_fs_80, params, cv_logreg_fs['mean_validation_score'][step])
    elif '100' in params:
        cv_logreg_fs_100 = fill_list(cv_logreg_fs_100, params, cv_logreg_fs['mean_validation_score'][step])
    step+=1
step = 0
for params in cv_logreg_na['params']:
    if '40' in params:
        cv_logreg_na_40 = fill_list(cv_logreg_na_40, params, cv_logreg_na['mean_validation_score'][step])
    elif '60' in params:
        cv_logreg_na_60 = fill_list(cv_logreg_na_60, params, cv_logreg_na['mean_validation_score'][step])
    elif '80' in params:
        cv_logreg_na_80 = fill_list(cv_logreg_na_80, params, cv_logreg_na['mean_validation_score'][step])
    elif '100' in params:
        cv_logreg_na_100 = fill_list(cv_logreg_na_100, params, cv_logreg_na['mean_validation_score'][step])
    step+=1

# create plot
fig, ax = plt.subplots()
index = np.arange(2)
bw=0.13
a=0.7
rects1 = plt.bar(index,cv_logreg_fs_40,bw,color='#63b3c2',label='C=40 w/ fs')
rects2 = plt.bar(index+bw,cv_logreg_fs_60,bw,color='#196fef',label='C=60 w/ fs')
rects3 = plt.bar(index+2*bw,cv_logreg_fs_80,bw,color='#0e1f60',label='C=80 w/ fs')
rects4 = plt.bar(index+3*bw,cv_logreg_fs_100,bw,color='#1df7ff',label='C=100 w/ fs')
rects5 = plt.bar(index+0.25*bw,cv_logreg_na_40,bw/2,alpha=a,color='r',label='w/o fs')
rects6 = plt.bar(index+1.25*bw,cv_logreg_na_60,bw/2,alpha=a,color='r')
rects5 = plt.bar(index+2.25*bw,cv_logreg_na_80,bw/2,alpha=a,color='r')
rects6 = plt.bar(index+3.25*bw,cv_logreg_na_100,bw/2,alpha=a,color='r')
plt.xlabel('solver')
plt.ylabel('mean validation score')
plt.title('Hyperparameter Results - Logistic Regression')
plt.xticks(index + 1.5*bw, ('lbfgs','saga'))
plt.ylim(0.860, 0.8625)
plt.legend()
plt.tight_layout()
plt.show
plt.savefig('data/Landsat/plots/LR_hyp_res.png')
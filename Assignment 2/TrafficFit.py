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
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score

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
tuned_parameters = {"BRidge": [{'alpha_1': np.logspace(2, -6, num=2), 
                                'lambda_1': np.logspace(2, -6, num=2), 
                                'fit_intercept': [True, False], 
                                'normalize':[True, False]}]}
score = 'neg_median_absolute_error' # 

# Models
models = {
    "BRidge": BayesianRidge()
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
            # continue your sklearn code here...
            start = time.time()
            clf = GridSearchCV(model, tuned_parameters[mkey], cv=kf, 
                               scoring=score)
            clf.fit(X_train, y_train)
            end = time.time()
            fitting_time = end - start
            
            print("Best parameters set found on train set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on train set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()
            
            cv_res = pd.DataFrame([str(item) for item in clf.cv_results_['params']], columns=["paras"])
            cv_res['rt_name'] = rt_name
            cv_res['sl_name'] = sl_name
            cv_res['method'] = mkey
            cv_res['mean_validation_score'] = means
            cv_res.sort_values(by='mean_validation_score', ascending=False, inplace=True)
            cv_res.reset_index(inplace=True,drop=True)
            cv_res.reset_index(inplace=True)
            cv.append(cv_res)
            
            print("Test Set Report:")
            print()
            best_model = clf.best_estimator_
            y_true, y_pred = y_test, best_model.predict(X_test)
            print("R-Square: the % of information explain by the fitted target variable: ")
            r2 = r2_score(y_true, y_pred)
            print(r2 * 100)
            print()
            print()
            out = {'method':mkey,'paras': str(best_model.get_params()),'metrics':r2, 'rt_name': rt_name,
                "sl_name":sl_name,"training_time": fitting_time, "NumOfFeatures": X_test.shape[1]
            }
            holdout.append(pd.DataFrame(out,index=[0]))

output_cv = pd.concat(cv)
output_ho = pd.concat(holdout)

output_ho.sort_values(inplace=True, ascending=False, by='metrics')

output_ho.sort_values(inplace=True, ascending=False, by='metrics')

output_cv.reset_index(inplace=True, drop=True)
output_ho.reset_index(inplace=True, drop=True)



output_cv.to_csv("data/Traffic_flow/cv_result.csv",index=False)
output_ho.to_csv("data/Traffic_flow/ho_result.csv",index=False)

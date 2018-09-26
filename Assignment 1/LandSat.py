# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 17:49:41 2018

@author: Junjie
"""

import time as time
import numpy as np
from __future__ import print_function
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

import pandas as pd

random_seed = 5432363

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

row_transform = {'Identity':row_identity ,
                'Max': row_max, 'MinMax': row_minmax
                 }
col_scalling = {'Identity':None ,'ZScore':  StandardScaler, 'MinMax': MinMaxScaler
                }

# Feature Extraction
feature_extract = {'PCA1': PCA(n_components=2L), 
                   'PCA2': PCA(n_components=0.95) ,
                   'PCA3': PCA(),
                   'BestK1': SelectKBest(f_regression, k=3), 'BestK2': SelectKBest(f_regression, k=6),
                   'BestK3': SelectKBest(f_regression, k=9), 'Identity': None 
}
# Cross Validation
skf  = StratifiedKFold(n_splits=5, random_state=random_seed)

# Hyperparameter Space
tuned_parameters = {
        "SVM": [{'kernel': ['rbf'], 
                     'C': np.logspace(-3, 3, num=7)},
                    {'kernel': ['linear'], 'C': np.logspace(-3, 3, num=7)}],
        "DT" :  [{'max_depth': [1,3,5,7,9,11]}],
        "Perceptron": [{'alpha': np.logspace(0, -5, num=5)}],
        "KNN": [{'n_neighbors': np.arange(2, 30, 6)}],
        "NB": [{'priors': [ (Y.value_counts()/len(Y)).values,  # use train as the prior distribution
                        [0.167, 0.167, 0.167, 0.167, 0.166 , 0.166] # uniform, no information prior
                        ]}]
}
score = 'f1_micro' # f1_macro

# Models
models = {
    "SVM": SVC(cache_size=9999999999)
    "NB": GaussianNB(), 
    "Perceptron": Perceptron(penalty = "l2"),
    "KNN": KNeighborsClassifier(algorithm='kd_tree'), 
    "DT": DecisionTreeClassifier()
}

# train/test 
cv = []
holdout = []
y_train0, X_train0 = extract(train)
y_test0, X_test0 = extract(test)
for rt_name, transformer in row_transform.iteritems():
    print("# Transforming row value by %s" % rt_name)
    X_train1 = transformer(X_train0)
    X_test1 = transformer(X_test0)
    for sl_name, scaler in col_scalling.iteritems():
        print("# Scaling column value by %s" % sl_name)
        if sl_name != 'Identity':
            sl = scaler()
            sl.fit(X_train1)
            X_train2 = sl.transform(X_train1)
            X_test2 = sl.transform(X_test1)
        else: 
            X_train2 = X_train1.copy()
            X_test2 = X_test1.copy()
        for fe_name, fe in feature_extract.iteritems():
            print("# Features Selection/Extraction value by %s" % fe_name)
            if fe_name in ['PCA1', 'PCA2']:
                fe.fit(X_train2)
                #print(fe.explained_variance_ratio_.cumsum())
                X_train = fe.transform(X_train2)
                X_test = fe.transform(X_test2)
            elif fe_name in ['BestK1']:
                fe.fit(X_train2, y_train0)
                #print(fe.get_support(indices=True))
                X_train = fe.transform(X_train2)
                X_test = fe.transform(X_test2)
            else: 
                X_train = X_train2.copy()
                X_test = X_test2.copy()   
            for mkey, model in models.iteritems():
                print("# Staring fittimg model of %s" % mkey)
                print()
                
                y_train = y_train0
                y_test = y_test0
                #y_train, X_train = extract(train)
                print("# Tuning hyper-parameters for %s" % score)
                print()
                start = time.time()
                clf = GridSearchCV(model, tuned_parameters[mkey], cv=skf, 
                                   scoring=score, n_jobs = -1)
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
                cv_res['fe_name'] = fe_name
                cv_res['method'] = mkey
                cv_res['mean_validation_score'] = means
                cv_res.sort_values(by='mean_validation_score', ascending=False, inplace=True)
                cv_res.reset_index(inplace=True,drop=True)
                cv_res.reset_index(inplace=True)
                cv.append(cv_res)
                
                print("Test Set Report:")
                print()
                #y_test, X_test = extract(test)
                best_model = clf.best_estimator_
                y_true, y_pred = y_test, best_model.predict(X_test)
                print("Presision, Recall, F1, Support by Class")
                print(classification_report(y_true, y_pred))
                print()
                print("Accuracy")
                accuracy = accuracy_score(y_true, y_pred)
                print("%0.02f" % (accuracy*100.))
                print()
                print("Confusion Matrix")
                print(confusion_matrix(y_true, y_pred))
                print()
                out = {'method':mkey,'paras': str(best_model.get_params()),'metrics':accuracy, 'rt_name': rt_name,
                    "sl_name":sl_name,"fe_name": fe_name,"training_time": fitting_time, "NumOfFeatures": X_test.shape[1]
                }
                holdout.append(pd.DataFrame(out,index=[0]))


output_cv = pd.concat(cv)
output_ho = pd.concat(holdout)

output_ho.sort_values(inplace=True, ascending=False, by='metrics')

output_ho.sort_values(inplace=True, ascending=False, by='metrics')

output_cv.reset_index(inplace=True, drop=True)
output_ho.reset_index(inplace=True, drop=True)



output_cv.to_csv("data/Landsat/cv_result.csv",index=False)
output_ho.to_csv("data/Landsat/ho_result.csv",index=False)





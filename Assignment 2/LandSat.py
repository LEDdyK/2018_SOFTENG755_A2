# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 17:49:41 2018

@author: Junjie/Lite
"""

import time as time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

print(__doc__)



if __name__ == "__main__":
    
    
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
    
    # Cross Validation: Stratified K Fold used for classification
    skf = StratifiedKFold(n_splits=5, random_state=random_seed)
    
    # Hyperparameter Space
    tuned_parameters = {
            "Logistics": [{'C':range(40, 101, 20),
                           'solver':['lbfgs', 'saga']}],
            "Neural": [{'hidden_layer_sizes':[(40,),(60,),(80,)],
                        'alpha': [0.0005, 0.001, 0.005, 0.01]}]
    }
    
    # define scoring
    score = 'f1_micro'
    
    # Models
    models = {
        "Logistics": LogisticRegression(multi_class='multinomial', max_iter=10000, random_state=random_seed),
        "Neural": MLPClassifier(max_iter=10000, random_state=random_seed)
    }
    
    # Matrix
    cv = []
    holdout = []
    
    # Split datasets into features (X) and outputs (y)
    y_train, X_train0 = extract(train)
    y_test, X_test0 = extract(test)
    
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
                X_train = sl.transform(X_train1)
                X_test = sl.transform(X_test1)
            else: 
                X_train = X_train1.copy()
                X_test = X_test1.copy()  
                
            for with_feat_sel in range(2):
                if with_feat_sel == 1:
                    # Feature Selection: Model based feature selection via recursion and cross validation
                    print("Choosing Features...")
                    select = RFECV(RandomForestClassifier(random_state=random_seed), cv=skf, scoring=score)
                    select.fit(X_train, y_train)
                    X_train_selected = select.transform(X_train)
                    X_test_selected = select.transform(X_test)
                    # display features selected via mask plot
                    mask = select.get_support()
                    plt.matshow(mask.reshape(1,-1), cmap='gray_r')
                    plt.xlabel('Index of Features')
                else:
                    print("trying without feature selection...")
                    X_train_selected = X_train
                    X_test_selected = X_test
                
                # apply the models to the data: mkey = key, model = value
                for mkey, model in models.items():
                    print("----------# Start fitting model of %s----------" % mkey)
                    # tune hyperparameters via GridSearchCV
                    print("# Tuning hyper-parameters for %s" % score)
                    time_Start = time.time()
                    clf = GridSearchCV(model, tuned_parameters[mkey], cv=skf, 
                                       scoring=score)
                    clf.fit(X_train_selected, y_train)
                    time_End = time.time()
                    fitting_time = time_End - time_Start
                    
                    print("Best hyperparameters set found on train set:")
                    print(clf.best_params_)
                    print("Grid scores on train set:")
                    means = clf.cv_results_['mean_test_score']
                    stds = clf.cv_results_['std_test_score']
                    
                    # store details of classification to cv_res
                    cv_res = pd.DataFrame([str(item) for item in clf.cv_results_['params']], columns=['params'])
                    cv_res['rt_name'] = rt_name
                    cv_res['sl_name'] = sl_name
                    cv_res['method'] = mkey
                    if with_feat_sel == 1:
                        cv_res['features'] = str(select.get_support(indices=True))
                    else:
                        cv_res['features'] = 'all'
                    cv_res['mean_validation_score'] = means
                    cv_res.sort_values(by='mean_validation_score', ascending=False, inplace=True)
                    cv_res.reset_index(inplace=True, drop=True)
                    cv.append(cv_res)
                    
                    # compare the results of the best model with the test set (true data)
                    print("Test Set Report:")
                    best_model = clf.best_estimator_
                    y_true, y_pred = y_test, best_model.predict(X_test_selected)
                    print("Accuracy classification score:")
                    acc = accuracy_score(y_true, y_pred)
                    print(acc * 100)
                    out = {'method':mkey,'params': str(best_model.get_params()),'metrics':acc, 'rt_name':rt_name,
                        'sl_name':sl_name,'training_time':fitting_time, 'NumOfFeatures':X_test_selected.shape[1], 
                        'Features - index': cv_res['features']
                    }
                    holdout.append(pd.DataFrame(out,index=[0]))
    
    # for outputting results
    output_cv = pd.concat(cv)
    output_cv.sort_values(inplace=True, ascending=False, by='mean_validation_score')
    output_cv.reset_index(inplace=True, drop=True)
    output_ho = pd.concat(holdout)
    output_ho.sort_values(inplace=True, ascending=False, by='metrics')
    output_ho.reset_index(inplace=True, drop=True)
    
    # save results to .csv file
    output_cv.to_csv("data/Landsat/cv_result.csv",index=False)
    output_ho.to_csv("data/Landsat/ho_result.csv",index=False)

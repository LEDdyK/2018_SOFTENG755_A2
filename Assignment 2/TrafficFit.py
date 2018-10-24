# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:48:35 2018

@author: Junjie/Lite
"""

import time as time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.linear_model import BayesianRidge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score

print(__doc__)



if __name__ == "__main__":
    
    random_seed = 755
    
    # Load data
    train = pd.read_csv("data/Traffic_flow/train.csv")
    test = pd.read_csv("data/Traffic_flow/test.csv")
    
    # Delete redundant cells
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
    
    # data transformation functions
    row_transform = {'Identity':row_identity}
    col_scaling = {'Identity':None ,'MinMax':MinMaxScaler}
    
    # Feature Extraction
    feature_extract = {'Quart_1': SelectKBest(f_regression, k=112), 
                       'Half': SelectKBest(f_regression, k=225),
                       'Quart_3': SelectKBest(f_regression, k=338), 
                       'All': None}
    
    # Cross Validation: K Fold used for regression
    kf = KFold(n_splits=5, random_state=random_seed)
    
    # Hyperparameter Space
    #From multiple tunings, the best models seemed to hold for the deafult alpha_1 
    #value and fit_intercept set to false.
    # Because we do not apply feature selection, we use the whole feature set thus
    #the model is prone to overfitting. To counter, we include a penalisation term, 
    #lambda_1 (default = 1^-6) of greater value to penalise the coefficient harder
    tuned_parameters = {"BRidge": [{'lambda_1': range(60, 201, 10)}]}
    
    # define scoring
    score = 'neg_mean_absolute_error' 
    
    # Models
    models = {
        "BRidge": BayesianRidge(fit_intercept=False, normalize=False)
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
                
            #for fe_name, fe in feature_extract.items():
            for fe_count in range(45, 451, 45):
                #if fe_name == 'All':
                if fe_count == 450:
                    print("Running with all features...")
                    X_train_selected = X_train
                    X_test_selected = X_test
                else:
                    # Feature Selection via f regression
                    print("Choosing %d features..." % fe_count)
                    fe = SelectKBest(f_regression, k=fe_count)
                    fe.fit(X_train, y_train)
                    X_train_selected = fe.transform(X_train)
                    X_test_selected = fe.transform(X_test)
                    # display features selected via mask plot
                    mask = fe.get_support()
                    plt.matshow(mask.reshape(1,-1), cmap='gray_r')
                    plt.xlabel('Index of Features')
                    
                # apply the regression to the data: mkey = key, model = value
                for mkey, model in models.items():
                    print("----------# Start fitting model of %s----------" % mkey)
                    # tune hyperparameters via GridSearchCV
                    print("# Tuning hyper-parameters for %s" % score)
                    time_Start = time.time()
                    reg = GridSearchCV(model, tuned_parameters[mkey], cv=kf, 
                                       scoring=score)
                    reg.fit(X_train_selected, y_train)
                    time_End = time.time()
                    fitting_time = time_End - time_Start
                    
                    print("Best hyperparameters set found on train set:")
                    print(reg.best_params_)
                    print("Grid scores on train set:")
                    means = reg.cv_results_['mean_test_score']
                    stds = reg.cv_results_['std_test_score']
                    
                    # store details of regression to cv_res
                    cv_res = pd.DataFrame([str(item) for item in reg.cv_results_['params']], columns=['params'])
                    cv_res['params'] = reg.cv_results_['params']
                    cv_res['rt_name'] = rt_name
                    cv_res['sl_name'] = sl_name
                    cv_res['method'] = mkey
                    cv_res['features'] = str(fe.get_support(indices=True))
                    cv_res['mean_validation_score'] = means
                    cv_res.sort_values(by='mean_validation_score', ascending=False, inplace=True)
                    cv_res.reset_index(inplace=True,drop=True)
                    cv.append(cv_res)
                    
                    # compare the results of the best model with the test set (true data)
                    print("Test Set Report:")
                    best_model = reg.best_estimator_
                    y_true, y_pred = y_test, best_model.predict(X_test_selected)
                    print("R-Square: the % of information explain by the fitted target variable: ")
                    r2 = r2_score(y_true, y_pred)
                    print(r2 * 100)
                    out = {'method':mkey,'paras': str(best_model.get_params()),'metrics':r2, 'rt_name': rt_name,
                        "sl_name":sl_name,"training_time": fitting_time, "NumOfFeatures": X_test_selected.shape[1],
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
    output_cv.to_csv("data/Traffic_flow/cv_result.csv",index=False)
    output_ho.to_csv("data/Traffic_flow/ho_result.csv",index=False)

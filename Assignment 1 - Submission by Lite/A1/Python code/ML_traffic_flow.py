# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 21:52:00 2018

@author: lkim564
"""

#from preprocessing demo
import category_encoders as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
#from linear regression demo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#read the dataset
traffic = pd.read_csv("traffic_flow_data.csv")

#PREPROCESSING
#describe the dataset
#traffic.describe()
#separating the data into features and targets
#traffic attributes - features
features = traffic.iloc[:,np.arange(traffic.shape[1]-1)].copy()
#segment 23(t+1) result - regression target
target = traffic.iloc[:,traffic.shape[1]-1].copy()
#create a class to select numerical or categorical columns 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

#initialize regression results dataframe
reg_res = pd.DataFrame(columns=['Feature', 'MSETrain', 'RSTrain', 'MSETest', 'RSTest'])

#establish loop paramenters
features_avail = features
for z in range (0, features_avail.shape[1]):
    feature_sel = features_avail[[features_avail.columns[z]]].copy()
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(list(feature_sel))),
        ('imputer', Imputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])
    full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline)
    ])
    
    #MODEL FITTING(linear regression)
    #readable model features
    feature_prepared = pd.DataFrame(data=full_pipeline.fit_transform(feature_sel),index=np.arange(1,7501))
    #split data into training and testing sets
    traffic_x_train, traffic_x_test, traffic_y_train, traffic_y_test = train_test_split(feature_prepared, target, test_size = 0.1, random_state = 1)
    #create linear regression object
    regr = linear_model.LinearRegression()
    #train the model using the training sets
    regr.fit(traffic_x_train, traffic_y_train)
    #make predictions on training targets using the training set against the trained model
    traffic_y_train_pred = regr.predict(traffic_x_train)
    #make predictions on testing targets using the testing set against the trained model
    traffic_y_test_pred = regr.predict(traffic_x_test)
    
    #MODEL VERIFICATION
    #results of regression - compare the predictions with actual results
    # lower Mean squared error = better, 0 = best
    # variance range = [0, 1], negative variance = trash
    MSETrain = mean_squared_error(traffic_y_train, traffic_y_train_pred)
    RSTrain = r2_score(traffic_y_train, traffic_y_train_pred)
    MSETest = mean_squared_error(traffic_y_test, traffic_y_test_pred)
    RSTest = r2_score(traffic_y_test, traffic_y_test_pred)
    reg_res = reg_res.append(pd.DataFrame([[feature_sel.columns[0], MSETrain, RSTrain, MSETest, RSTest]], 
                                          columns = ['Feature', 'MSETrain', 'RSTrain', 'MSETest', 'RSTest']), ignore_index=True)
#for comparing minimums
ref_MSE = reg_res['MSETest'].min()
#for checking available features
ref_feat = reg_res.iloc[reg_res['MSETest'].idxmin()]['Feature']
features_saved = features_avail[[ref_feat]].copy()
features_avail = features_avail.drop([ref_feat], axis=1, inplace=False)
cond = True
loop_count = 0

#SELECT FEATURES
while (features_avail.shape[1] > 0 and cond):
    #keep track of number of iterations
    loop_count += 1
    print(loop_count)
    #find the best combination of features
    reg_res = pd.DataFrame(columns=['Feature', 'MSETrain', 'RSTrain', 'MSETest', 'RSTest'])
    for y in range (0, features_avail.shape[1]):
        feature_sel = features_saved.join(features_avail[[features_avail.columns[y]]].copy())
        num_pipeline = Pipeline([
            ('selector', DataFrameSelector(list(feature_sel))),
            ('imputer', Imputer(strategy='median')),
            ('std_scaler', StandardScaler())
        ])
        full_pipeline = FeatureUnion(transformer_list=[
            ('num_pipeline', num_pipeline)
        ])
        
        #MODEL FITTING(linear regression)
        feature_prepared = pd.DataFrame(data=full_pipeline.fit_transform(feature_sel),index=np.arange(1,7501))
        traffic_x_train, traffic_x_test, traffic_y_train, traffic_y_test = train_test_split(feature_prepared, target, test_size = 0.1, random_state = 1)
        regr = linear_model.LinearRegression()
        regr.fit(traffic_x_train, traffic_y_train)
        traffic_y_train_pred = regr.predict(traffic_x_train)
        traffic_y_test_pred = regr.predict(traffic_x_test)
        
        #MODEL VERIFICATION
        MSETrain = mean_squared_error(traffic_y_train, traffic_y_train_pred)
        RSTrain = r2_score(traffic_y_train, traffic_y_train_pred)
        MSETest = mean_squared_error(traffic_y_test, traffic_y_test_pred)
        RSTest = r2_score(traffic_y_test, traffic_y_test_pred)
        reg_res = reg_res.append(pd.DataFrame([[feature_sel.columns[feature_sel.shape[1]-1], MSETrain, RSTrain, MSETest, RSTest]], 
                                              columns = ['Feature', 'MSETrain', 'RSTrain', 'MSETest', 'RSTest']), ignore_index=True)
    #make comparisons and check if an optimal model has been achieved
    cur_MSE = reg_res['MSETest'].min()
    #a new model is better than previous model (comparing Mean Squared Errors) - save values and reiterate
    if (ref_MSE >= cur_MSE):
        ref_MSE = cur_MSE
        ref_feat = reg_res.iloc[reg_res['MSETest'].idxmin()]['Feature']
        features_saved = features_saved.join(features_avail[[ref_feat]].copy())
        features_avail = features_avail.drop([ref_feat], axis=1, inplace=False)
    #previous model is better than all new models - escape loop
    else:
        cond = False
        

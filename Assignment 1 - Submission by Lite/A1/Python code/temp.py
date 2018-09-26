# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
worldcup = pd.read_csv("2018 worldcup.csv", index_col = 0)

#PREPROCESSING
#describe the dataset
worldcup.describe()
#separating the data into features and targets
#world cup attributes - features
w_features = worldcup.iloc[:,np.arange(28)].copy()
#world cup goal result - regression target
w_goals = worldcup.iloc[:,28].copy()
#world cup match result - classification target
w_results = worldcup.iloc[:,29].copy()
# Create a class to select numerical or categorical columns 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

"""
#SELECT FEATURES for test_size = 0.1
#w_features_num: numerical features
w_features_num = w_features.drop(['Date', 'Location', 'Phase', 'Team1', 'Team2', 'Team1_Continent', 'Team2_Continent', 'Normal_Time', 
                                  'Team2_Corners', 'Team2_Pass_Accuracy(%)', 'Team1_Corners', 'Team1_Attempts', 'Team2_Offsides', 'Team1_Pass_Accuracy(%)', 'Team2_Fouls'], axis=1, inplace=False)
#w_features_cat: categorical features
w_features_cat = w_features[['Team1_Continent']].copy()
#must have (default) features
w_features_def_num = w_features[['Team2_Corners', 'Team2_Pass_Accuracy(%)', 'Team1_Corners', 'Team1_Attempts', 'Team2_Offsides', 'Team1_Pass_Accuracy(%)', 'Team2_Fouls']].copy()
w_features_def_cat = w_features[['Phase', 'Normal_Time', 'Team2_Continent', 'Location', 'Date', 'Team1', 'Team2']].copy()
"""
#SELECT FEATURES for test_size = 0.2
#w_features_num: numerical features
w_features_num = w_features.drop(['Date', 'Location', 'Phase', 'Team1', 'Team2', 'Team1_Continent', 'Team2_Continent', 'Normal_Time', 
                                  'Team1_Distance_Covered', 'Team1_Ball_Possession(%)', 'Team2_Attempts', 'Team2_Corners', 'Team2_Ball_Possession(%)', 'Team1_Attempts', 'Team2_Fouls', 'Team1_Red_Card', 'Team1_Ball_Recovered'], axis=1, inplace=False)
#w_features_cat: categorical features
w_features_cat = w_features[['Phase', 'Team1_Continent', 'Normal_Time']].copy()
#must have (default) features
w_features_def_num = w_features[['Team1_Distance_Covered', 'Team1_Ball_Possession(%)', 'Team2_Attempts', 'Team2_Corners', 'Team2_Ball_Possession(%)', 'Team1_Attempts', 'Team2_Fouls', 'Team1_Red_Card', 'Team1_Ball_Recovered']].copy()
w_features_def_cat = w_features[['Team2_Continent', 'Location', 'Date', 'Team1', 'Team2']].copy()

#initialize regression results dataframe
reg_res = pd.DataFrame(columns=['Feature', 'MSETrain', 'RSTrain', 'MSETest', 'RSTest'])

#DEFAULT REGRESSION
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(list(w_features_def_num))),
        ('imputer', Imputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(list(w_features_def_cat))),
        ('cat_encoder', cs.OneHotEncoder(drop_invariant=True))
    ])
full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])
#MODEL FITTING (linear regression)
#readable model features
feature_prepared = pd.DataFrame(data=full_pipeline.fit_transform(w_features_def_num.join(w_features_def_cat)),index=np.arange(1, 65))
#split data into training and testing sets
worldcup_x_train, worldcup_x_test, worldcup_y_train, worldcup_y_test = train_test_split(feature_prepared, w_goals, test_size = 0.2, random_state = 0)
#create linear regression object
regr = linear_model.LinearRegression()
#train the model using the training sets
regr.fit(worldcup_x_train, worldcup_y_train)
#make predictions on training targets using the training set against the trained model
worldcup_y_train_pred = regr.predict(worldcup_x_train)
#make predictions on testing targets using the testing set against the trained model
worldcup_y_test_pred = regr.predict(worldcup_x_test)
#MODEL VERIFICATION
#results of regression - compare the predictions with actual results
# lower Mean squared error = better, 0 = best
# variance range = [0, 1], negative variance = trash
MSETrain = mean_squared_error(worldcup_y_train, worldcup_y_train_pred)
RSTrain = r2_score(worldcup_y_train, worldcup_y_train_pred)
MSETest = mean_squared_error(worldcup_y_test, worldcup_y_test_pred)
RSTest = r2_score(worldcup_y_test, worldcup_y_test_pred)
reg_res = reg_res.append(pd.DataFrame([['Default', MSETrain, RSTrain, MSETest, RSTest]], 
                                      columns = ['Feature', 'MSETrain', 'RSTrain', 'MSETest', 'RSTest']), ignore_index=True)

#initialize regression results dataframe
ridge_reg_res = pd.DataFrame(columns=['Alpha', 'MSETrain', 'RSTrain', 'MSETest', 'RSTest'])

from sklearn.linear_model import Ridge

for z in range (0, 100):
    setting = float(z)
    ridge_reg = Ridge(alpha = setting/1000)
    ridge_reg.fit(worldcup_x_train, worldcup_y_train)
    ridge_worldcup_y_train_pred = ridge_reg.predict(worldcup_x_train)
    ridge_worldcup_y_test_pred = ridge_reg.predict(worldcup_x_test)
    ridge_MSETrain = mean_squared_error(worldcup_y_train, ridge_worldcup_y_train_pred)
    ridge_RSTrain = r2_score(worldcup_y_train, ridge_worldcup_y_train_pred)
    ridge_MSETest = mean_squared_error(worldcup_y_test, ridge_worldcup_y_test_pred)
    ridge_RSTest = r2_score(worldcup_y_test, ridge_worldcup_y_test_pred)
    ridge_reg_res = ridge_reg_res.append(pd.DataFrame([[setting/1000, ridge_MSETrain, ridge_RSTrain, ridge_MSETest, ridge_RSTest]], 
                                                columns = ['Alpha', 'MSETrain', 'RSTrain', 'MSETest', 'RSTest']), ignore_index=True)

"""
#FEATURE SELECTION
for z in range (0, w_features_num.shape[1] + w_features_cat.shape[1]):
    #add features to set
    if (z < w_features_num.shape[1]):
        w_features_select_num = w_features[[w_features_num.columns[z]]].copy()
        w_features_select_num = w_features_select_num.join(w_features_def_num)
        num_pipeline = Pipeline([
                ('selector', DataFrameSelector(list(w_features_select_num))),
                ('imputer', Imputer(strategy='median')),
                ('std_scaler', StandardScaler())
            ])
        cat_pipeline = Pipeline([
                ('selector', DataFrameSelector(list(w_features_def_cat))),
                ('cat_encoder', cs.OneHotEncoder(drop_invariant=True))
            ])
        full_pipeline = FeatureUnion(transformer_list=[
            ('num_pipeline', num_pipeline),
            ('cat_pipeline', cat_pipeline)
        ])
        features_used = w_features_select_num.join(w_features_def_cat)
    else:
        w_features_select_cat = w_features[[w_features_cat.columns[z - w_features_num.shape[1]]]].copy()
        w_features_select_cat = w_features_select_cat.join(w_features_def_cat)
        num_pipeline = Pipeline([
                ('selector', DataFrameSelector(list(w_features_def_num))),
                ('imputer', Imputer(strategy='median')),
                ('std_scaler', StandardScaler())
            ])
        cat_pipeline = Pipeline([
                ('selector', DataFrameSelector(list(w_features_select_cat))),
                ('cat_encoder', cs.OneHotEncoder(drop_invariant=True))
            ])
        full_pipeline = FeatureUnion(transformer_list=[
            ('num_pipeline', num_pipeline),
            ('cat_pipeline', cat_pipeline)
        ])
        features_used = w_features_def_num.join(w_features_select_cat)
    
    #MODEL FITTING(linear regression)
    feature_prepared = pd.DataFrame(data=full_pipeline.fit_transform(features_used),index=np.arange(1,65))
    worldcup_cleaned = pd.concat([feature_prepared, w_goals.to_frame(), w_results.to_frame()], axis=1)
    worldcup_x_train, worldcup_x_test, worldcup_y_train, worldcup_y_test = train_test_split(feature_prepared, w_goals, test_size = 0.2, random_state = 0)
    regr = linear_model.LinearRegression()
    regr.fit(worldcup_x_train, worldcup_y_train)
    worldcup_y_train_pred = regr.predict(worldcup_x_train)
    worldcup_y_test_pred = regr.predict(worldcup_x_test)

    #MODEL VERIFICATION
    MSETrain = mean_squared_error(worldcup_y_train, worldcup_y_train_pred)
    RSTrain = r2_score(worldcup_y_train, worldcup_y_train_pred)
    MSETest = mean_squared_error(worldcup_y_test, worldcup_y_test_pred)
    RSTest = r2_score(worldcup_y_test, worldcup_y_test_pred)
    if (z < w_features_num.shape[1]):
        reg_res = reg_res.append(pd.DataFrame([[w_features_select_num.columns[0], MSETrain, RSTrain, MSETest, RSTest]], 
                                              columns = ['Feature', 'MSETrain', 'RSTrain', 'MSETest', 'RSTest']), ignore_index=True)
    else:
        reg_res = reg_res.append(pd.DataFrame([[w_features_select_cat.columns[0], MSETrain, RSTrain, MSETest, RSTest]], 
                                          columns = ['Feature', 'MSETrain', 'RSTrain', 'MSETest', 'RSTest']), ignore_index=True)
"""
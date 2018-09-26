# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 07:33:50 2018

@author: KiwiDivo
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
#traffic attributes - features
features = traffic.iloc[:,np.arange(traffic.shape[1]-1)].copy()
#segment 23(t+1) result - regression target
target = traffic.iloc[:,traffic.shape[1]-1].copy()

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

feature_sel = features[['Segment_22(t)', 
                        'Segment_27(t)', 
                        'Segment_16(t)', 
                        'Segment_20(t-1)', 
                        'Segment_23(t)', 
                        'Segment_19(t-1)', 
                        'Segment_10(t-6)', 
                        'Segment_27(t-5)', 
                        'Segment_19(t-8)', 
                        'Segment_19(t-2)', 
                        'Segment_9(t-6)', 
                        'Segment_20(t-2)', 
                        'Segment_29(t)', 
                        'Segment_34(t)', 
                        'Segment_38(t-5)', 
                        'Segment_37(t-3)', 
                        'Segment_37(t)', 
                        'Segment_42(t-5)', 
                        'Segment_25(t-1)', 
                        'Segment_2(t)', 
                        'Segment_10(t-8)', 
                        'Segment_5(t)', 
                        'Segment_4(t-7)', 
                        'Segment_38(t)', 
                        'Segment_17(t-1)', 
                        'Segment_35(t-1)', 
                        'Segment_34(t-1)', 
                        'Segment_27(t-1)', 
                        'Segment_15(t)', 
                        'Segment_29(t-2)', 
                        'Segment_9(t-1)', 
                        'Segment_5(t-5)', 
                        'Segment_21(t)', 
                        'Segment_31(t)', 
                        'Segment_17(t-9)', 
                        'Segment_15(t-9)', 
                        'Segment_14(t-9)', 
                        'Segment_41(t-5)', 
                        'Segment_43(t-4)', 
                        'Segment_38(t-3)', 
                        'Segment_10(t-4)', 
                        'Segment_10(t-5)', 
                        'Segment_39(t-3)', 
                        'Segment_7(t-9)', 
                        'Segment_9(t-8)', 
                        'Segment_14(t-8)', 
                        'Segment_17(t-7)', 
                        'Segment_19(t-6)', 
                        'Segment_35(t-4)', 
                        'Segment_29(t-4)', 
                        'Segment_17(t)', 
                        'Segment_10(t-3)', 
                        'Segment_16(t-9)', 
                        'Segment_16(t-7)', 
                        'Segment_34(t-9)', 
                        'Segment_32(t-8)', 
                        'Segment_35(t-9)', 
                        'Segment_33(t-9)', 
                        'Segment_32(t-4)', 
                        'Segment_33(t-4)', 
                        'Segment_5(t-9)', 
                        'Segment_11(t-6)', 
                        'Segment_33(t-6)', 
                        'Segment_32(t-6)', 
                        'Segment_33(t-7)', 
                        'Segment_30(t-7)', 
                        'Segment_19(t-9)', 
                        'Segment_41(t-6)', 
                        'Segment_38(t-4)', 
                        'Segment_5(t-6)', 
                        'Segment_28(t-9)', 
                        'Segment_9(t-9)', 
                        'Segment_42(t-1)', 
                        'Segment_38(t-6)', 
                        'Segment_30(t-1)', 
                        'Segment_15(t-8)', 
                        'Segment_27(t-9)', 
                        'Segment_26(t-9)', 
                        'Segment_27(t-3)', 
                        'Segment_38(t-2)', 
                        'Segment_37(t-2)', 
                        'Segment_42(t-3)', 
                        'Segment_10(t)', 
                        'Segment_11(t)', 
                        'Segment_29(t-4)', 
                        'Segment_34(t-6)', 
                        'Segment_1(t-2)', 
                        'Segment_14(t-1)', 
                        'Segment_40(t-5)', 
                        'Segment_6(t-5)', 
                        'Segment_8(t-5)', 
                        'Segment_9(t-5)', 
                        'Segment_8(t-3)', 
                        'Segment_15(t-3)', 
                        'Segment_1(t-4)', 
                        'Segment_15(t-5)', 
                        'Segment_31(t-3)', 
                        'Segment_6(t-4)', 
                        'Segment_7(t-5)', 
                        'Segment_6(t-7)', 
                        'Segment_7(t-6)', 
                        'Segment_38(t-7)', 
                        'Segment_14(t-6)', 
                        'Segment_30(t-4)', 
                        'Segment_32(t-5)', 
                        'Segment_43(t-5)', 
                        'Segment_15(t-4)', 
                        'Segment_30(t-9)', 
                        'Segment_29(t-9)', 
                        'Segment_8(t-1)', 
                        'Segment_8(t-6)', 
                        'Segment_4(t-6)', 
                        'Segment_2(t-6)', 
                        'Segment_44(t-5)', 
                        'Segment_31(t-9)', 
                        'Segment_31(t-4)', 
                        'Segment_5(t-8)', 
                        'Segment_9(t-3)', 
                        'Segment_18(t-1)', 
                        'Segment_18(t-9)', 
                        'Segment_45(t-5)', 
                        'Segment_18(t-7)']].copy()
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
"""
#split data into training and testing sets
traffic_x_train, traffic_x_test, traffic_y_train, traffic_y_test = train_test_split(feature_prepared, target, test_size = 0.1, random_state = 0)
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
reg_res = pd.DataFrame([[feature_sel.columns[0], MSETrain, RSTrain, MSETest, RSTest]], 
                       columns = ['Feature', 'MSETrain', 'RSTrain', 'MSETest', 'RSTest'])

print('Coefficients and Intercept are: ', regr.coef_,"   ",regr.intercept_,' respectively')

#initialize regression results dataframe
ridge_reg_res = pd.DataFrame(columns=['Alpha', 'MSETrain', 'RSTrain', 'MSETest', 'RSTest'])
"""

from sklearn.linear_model import Ridge
#cross validation
from sklearn.cross_validation import cross_val_score
"""
for z in range (0, 100):
    setting = float(z)
    ridge_reg = Ridge(alpha = setting/1000)
    ridge_reg.fit(traffic_x_train, traffic_y_train)
    ridge_traffic_y_train_pred = ridge_reg.predict(traffic_x_train)
    ridge_traffic_y_test_pred = ridge_reg.predict(traffic_x_test)
    ridge_MSETrain = mean_squared_error(traffic_y_train, ridge_traffic_y_train_pred)
    ridge_RSTrain = r2_score(traffic_y_train, ridge_traffic_y_train_pred)
    ridge_MSETest = mean_squared_error(traffic_y_test, ridge_traffic_y_test_pred)
    ridge_RSTest = r2_score(traffic_y_test, ridge_traffic_y_test_pred)
    ridge_reg_res = ridge_reg_res.append(pd.DataFrame([[setting/1000, ridge_MSETrain, ridge_RSTrain, ridge_MSETest, ridge_RSTest]], 
                                                columns = ['Alpha', 'MSETrain', 'RSTrain', 'MSETest', 'RSTest']), ignore_index=True)
    
use_feat = num_feat.join(cat_feat)
"""
"""
save_score = pd.DataFrame(columns=['Alpha', 'MSE'])
for z in range(1, 201):
    #MODEL FITTING(linear regression)
    regr = linear_model.Ridge(alpha = 8.4 + float(z)/100)
    scores = cross_val_score(regr, feature_prepared, target, cv = 10, scoring = 'neg_mean_squared_error')
    save_score = save_score.append(pd.DataFrame([[8.4 + float(z)/100, scores.mean()]], 
                                                columns = ['Alpha', 'MSE']), ignore_index = True)
    print(z)
"""
regr = linear_model.LinearRegression()
regr.fit(feature_prepared, target)
pred = regr.predict(feature_prepared)
plt.scatter(list(range(1,7501)), target - pred, s = 4)
plt.show()
print(regr.score(feature_prepared, target))

regr = linear_model.Ridge(alpha = 8.5)
regr.fit(feature_prepared, target)
pred = regr.predict(feature_prepared)
plt.scatter(list(range(1,7501)), target - pred, s = 4)
#plt.scatter(list(range(1,7501)), target, s = 200, marker = (5, 0))
#plt.scatter(list(range(1,7501)), pred, s = 30)
plt.show()
print(regr.score(feature_prepared, target))
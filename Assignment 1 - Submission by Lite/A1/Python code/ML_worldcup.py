# -*- coding: utf-8 -*-
"""


@author: KiwiDivo
"""

#from preprocessing demo
import category_encoders as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
#from linear regression demo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#DECISION TREE CLASSIFICATION
from sklearn import tree
#NEAREST NEIGHBOUR CLASSIFICATION
from sklearn.neighbors import KNeighborsClassifier
#NAIVE BAYES CLASSIFICATION
from sklearn.naive_bayes import GaussianNB
#SVM CLASSIFICATION
from sklearn.svm import SVC
#PERCEPTRON CLASSIFICATION
from sklearn.linear_model import Perceptron
#cross validation
from sklearn.cross_validation import cross_val_score

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
    
#numerical features
num_feat = w_features[['Team1_Attempts', 'Team2_Attempts', 'Team1_Corners', 'Team2_Corners', 
                       'Team1_Offsides', 'Team2_Offsides', 'Team1_Ball_Possession(%)', 'Team2_Ball_Possession(%)', 
                       'Team1_Pass_Accuracy(%)', 'Team2_Pass_Accuracy(%)', 'Team1_Distance_Covered', 'Team2_Distance_Covered', 
                       'Team1_Ball_Recovered', 'Team2_Ball_Recovered', 'Team1_Yellow_Card',  'Team2_Yellow_Card', 
                       'Team1_Red_Card', 'Team2_Red_Card', 'Team1_Fouls', 'Team2_Fouls']].copy()
#categorical features - remove(Date, Location, Phase, Normal_Time)
cat_feat = w_features[['Team1_Continent', 'Team2_Continent']].copy()
#default feature set
def_cat = w_features[['Team1', 'Team2']].copy()

cond = True
num_ex = False

while ((num_feat.shape[1] + cat_feat.shape[1]) > 0 and cond):
    save_score = pd.DataFrame(columns = ['Feature1', 'Feature2', 'MSE'])
    #finding the optimum set of features
    for z in range(0, num_feat.shape[1] + cat_feat.shape[1] + 1, 2):
        #add features to set
        if (z < num_feat.shape[1]):
            sel_num = w_features[[num_feat.columns[z], num_feat.columns[z + 1]]].copy()
            new_feat = sel_num
            if (num_ex):
                sel_num = sel_num.join(def_num)
            num_pipeline = Pipeline([
                ('selector', DataFrameSelector(list(sel_num))),
                ('imputer', Imputer(strategy='median')),
                ('std_scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ('selector', DataFrameSelector(list(def_cat))),
                ('cat_encoder', cs.OneHotEncoder(drop_invariant=True))
            ])
            full_pipeline = FeatureUnion(transformer_list=[
                ('num_pipeline', num_pipeline),
                ('cat_pipeline', cat_pipeline)
            ])
            use_feat = sel_num.join(def_cat)
        else:
            if (z < num_feat.shape[1] + cat_feat.shape[1]):
                sel_cat = w_features[[cat_feat.columns[z - num_feat.shape[1]], cat_feat.columns[z - num_feat.shape[1] + 1]]].copy()
                new_feat = sel_cat
                sel_cat = sel_cat.join(def_cat)
            else:
                sel_cat = def_cat
                new_feat = sel_cat
            cat_pipeline = Pipeline([
                ('selector', DataFrameSelector(list(sel_cat))),
                ('cat_encoder', cs.OneHotEncoder(drop_invariant=True))
            ])
            if (num_ex):
                num_pipeline = Pipeline([
                    ('selector', DataFrameSelector(list(def_num))),
                    ('imputer', Imputer(strategy='median')),
                    ('std_scaler', StandardScaler())
                ])
                full_pipeline = FeatureUnion(transformer_list=[
                    ('num_pipeline', num_pipeline),
                    ('cat_pipeline', cat_pipeline)
                ])
                use_feat = def_num.join(sel_cat)
            else:
                full_pipeline = FeatureUnion(transformer_list=[
                    ('cat_pipeline', cat_pipeline)
                ])
                use_feat = sel_cat
        
        feat_prep = pd.DataFrame(data = full_pipeline.fit_transform(use_feat), index=np.arange(1,65))
        
        """
        #MODEL FITTING(linear regression)
        regr = linear_model.LinearRegression()
        #MODEL FITTING(ridge regression)
        #regr = linear_model.Ridge()
        scores = cross_val_score(regr, feat_prep, w_goals, cv = 3, scoring = 'neg_mean_squared_error')
        save_score = save_score.append(pd.DataFrame([[new_feat.columns[0], new_feat.columns[1], scores.mean()]], 
                                                    columns = ['Feature1', 'Feature2', 'MSE']), ignore_index = True)
        """
        #MODEL FITTING(classification)
        #DECISION TREE
        #classifier = tree.DecisionTreeClassifier()
        #NEAREST NEIGHTBOUR
        #classifier = KNeighborsClassifier()
        #NAIVE BAYES
        #classifier = GaussianNB()
        #SVM
        #classifier = SVC()
        #Perceptron
        classifier = Perceptron()
        scores = cross_val_score(classifier, feat_prep, w_results, cv = 3, scoring = 'accuracy')
        save_score = save_score.append(pd.DataFrame([[new_feat.columns[0], new_feat.columns[1], scores.mean()]], 
                                                    columns = ['Feature1', 'Feature2', 'MSE']), ignore_index = True)
        
    
    if (scores.mean() == save_score['MSE'].max()):
        cond = False
    else:
        best_sc = save_score.loc[save_score['MSE'] == save_score['MSE'].max()] 
        add_featA = best_sc.iloc[0]['Feature1']
        add_featB = best_sc.iloc[0]['Feature2']
        if (add_featA == 'Team1_Continent'):
            def_cat = def_cat.join(cat_feat[['Team1_Continent']].copy())
            def_cat = def_cat.join(cat_feat[['Team2_Continent']].copy())
            cat_feat = cat_feat.drop(['Team1_Continent', 'Team2_Continent'], axis=1, inplace=False)
        else:
            if (not num_ex):
                def_num = num_feat[[add_featA, add_featB]].copy()
                num_ex = True
            else:
                def_num = def_num.join(num_feat[[add_featA]].copy())
                def_num = def_num.join(num_feat[[add_featB]].copy())
            num_feat = num_feat.drop([add_featA, add_featB], axis=1, inplace=False)


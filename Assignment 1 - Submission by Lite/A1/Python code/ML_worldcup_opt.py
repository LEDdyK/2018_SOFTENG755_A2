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

  
#regression features
#numerical features (ridge)
num_feat = w_features[['Team1_Ball_Recovered', 'Team2_Ball_Recovered']].copy()
#categorical features - remove(Date, Location, Phase, Normal_Time) (ridge)
cat_feat = w_features[['Team1', 'Team2']].copy()
#numerical features (regular)
#num_feat = w_features[['Team1_Fouls', 'Team1_Ball_Recovered', 'Team1_Attempts', 
#                       'Team2_Fouls', 'Team2_Ball_Recovered', 'Team2_Attempts']].copy()
#categorical features - remove(Date, Location, Phase, Normal_Time) (regular)
#cat_feat = w_features[['Team1', 'Team1_Continent', 'Team2', 'Team2_Continent']].copy()

#classification features
#decision tree features
#num_feat = w_features[['Team1_Red_Card', 'Team2_Red_Card']].copy()
#cat_feat = w_features[['Team1', 'Team2']].copy()
#nearest neighbour features
#num_feat = w_features[['Team1_Ball_Possession(%)', 'Team2_Ball_Possession(%)']].copy()
#cat_feat = w_features[['Team1', 'Team2']].copy()
#naive bayes features
#num_feat = w_features[['Team1_Red_Card', 'Team2_Red_Card']].copy()
#cat_feat = w_features[['Team1', 'Team2', 'Team1_Continent', 'Team2_Continent']].copy()
#svm features
#num_feat = w_features[['Team1_Yellow_Card', 'Team2_Yellow_Card', 'Team1_Pass_Accuracy(%)', 'Team2_Pass_Accuracy(%)']].copy()
#cat_feat = w_features[['Team1', 'Team2']].copy()
#perceptron features
#num_feat = w_features[['Team1_Yellow_Card', 'Team2_Yellow_Card', 'Team1_Corners', 'Team2_Corners', 
#                       'Team1_Ball_Possession(%)', 'Team2_Ball_Possession(%)']].copy()
#cat_feat = w_features[['Team1', 'Team2', 'Team1_Continent', 'Team2_Continent']].copy()



num_pipeline = Pipeline([
    ('selector', DataFrameSelector(list(num_feat))),
    ('imputer', Imputer(strategy='median')),
    ('std_scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(list(cat_feat))),
    ('cat_encoder', cs.OneHotEncoder(drop_invariant=True))
])
full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])
use_feat = num_feat.join(cat_feat)
"""
save_score = pd.DataFrame(columns=['Alpha', 'MSE'])
for z in range(0, 21):
    #MODEL FITTING(linear regression)
    feat_prep = pd.DataFrame(data = full_pipeline.fit_transform(use_feat), index=np.arange(1,65))
    regr = linear_model.Ridge(alpha = z)
    scores = cross_val_score(regr, feat_prep, w_goals, cv = 3, scoring = 'r2')
    save_score = save_score.append(pd.DataFrame([[z, scores.mean()]], 
                                                columns = ['Alpha', 'MSE']), ignore_index = True)
    print(z)
"""
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
#HYPERPARAMETER TUNING and CROSS VALIDATION (KFOLDS)
from sklearn.grid_search import GridSearchCV

feat_prep = pd.DataFrame(data = full_pipeline.fit_transform(use_feat), index=np.arange(1,65))
"""
#DECISION TREE
classifier = tree.DecisionTreeClassifier()
mdrange = list(range(1, 21))
mslrange = list(range(1, 21))
param_grid = dict(max_depth = mdrange, min_samples_leaf = mslrange)
#NEAREST NEIGHBOUR
classifier = KNeighborsClassifier()
nrange = list(range(1, 41))
woptions = ['uniform', 'distance']
param_grid = dict(n_neighbors = nrange, weights = woptions)
#NAIVE BAYES
classifier = GaussianNB()
scores = cross_val_score(classifier, feat_prep, w_results, cv = 3, scoring = 'accuracy')
#SVM
classifier = SVC()
koptions = ['linear', 'poly', 'rbf', 'sigmoid']
crange = [2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3]
param_grid = dict(kernel = koptions, C = crange)
#PERCEPTRON
classifier = Perceptron()
aoptions = [1**-4, 1**-3, 1**-2, 1**-1, 1, 10, 100, 1000]
param_grid = dict(alpha = aoptions)

grid = GridSearchCV(classifier, param_grid, cv = 3, scoring = 'accuracy')
scores = grid.fit(feat_prep, w_results)

results = scores.grid_scores_
print(' ')
print(scores.best_score_)
print(' ')
print(scores.best_params_)
print(' ')
print(scores.best_estimator_)"""

"""
#generate tables and plots
#regular regression
regression = linear_model.LinearRegression()
regression.fit(feat_prep, w_goals)
wc = regression.coef_
wi = regression.intercept_
pred = regression.predict(feat_prep)
acc = mean_squared_error(w_goals, pred)
"""
#ridge regression
regression = linear_model.Ridge(alpha = 15.7)
regression.fit(feat_prep, w_goals)
wc = regression.coef_
wi = regression.intercept_
pred = regression.predict(feat_prep)
acc = mean_squared_error(w_goals, pred)
plt.scatter(list(range(1,65)), w_goals, s = 200, marker = (5, 0))
plt.scatter(list(range(1,65)), pred, s = 30)
plt.show()
print(regression.score(feat_prep, w_goals))
"""
#DECISION TREE
classifier = tree.DecisionTreeClassifier(max_depth = 8, min_samples_leaf = 1)
classifier.fit(feat_prep, w_results)
pred = pd.DataFrame(classifier.predict(feat_prep))
pred = pred.iloc[:,0].copy()
plt.scatter(list(range(1,65)), w_results, s = 200, marker = (5, 0))
plt.scatter(list(range(1,65)), pred, s = 30)
plt.show()
#NEAREST NEIGHBOUR
classifier = KNeighborsClassifier(n_neighbors = 4, weights = 'uniform')
classifier.fit(feat_prep, w_results)
pred = pd.DataFrame(classifier.predict(feat_prep))
pred = pred.iloc[:,0].copy()
plt.scatter(list(range(1,65)), w_results, s = 200, marker = (5, 0))
plt.scatter(list(range(1,65)), pred, s = 30)
plt.show()
#NAIVE BAYES
classifier = GaussianNB()
classifier.fit(feat_prep, w_results)
pred = pd.DataFrame(classifier.predict(feat_prep))
pred = pred.iloc[:,0].copy()
plt.scatter(list(range(1,65)), w_results, s = 200, marker = (5, 0))
plt.scatter(list(range(1,65)), pred, s = 30)
plt.show()
#SVM
classifier = SVC(C = 1, kernel = 'rbf')
classifier.fit(feat_prep, w_results)
wi = classifier.intercept_
pred = pd.DataFrame(classifier.predict(feat_prep))
pred = pred.iloc[:,0].copy()
plt.scatter(list(range(1,65)), w_results, s = 200, marker = (5, 0))
plt.scatter(list(range(1,65)), pred, s = 30)
plt.show()
#PERCEPTRON
classifier = Perceptron(alpha = 1.0)
classifier.fit(feat_prep, w_results)
wc = classifier.coef_
wi = classifier.intercept_
pred = pd.DataFrame(classifier.predict(feat_prep))
pred = pred.iloc[:,0].copy()
plt.scatter(list(range(1,65)), w_results, s = 200, marker = (5, 0))
plt.scatter(list(range(1,65)), pred, s = 30)
plt.show()
"""
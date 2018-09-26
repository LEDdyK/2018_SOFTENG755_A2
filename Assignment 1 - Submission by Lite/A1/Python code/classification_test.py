# -*- coding: utf-8 -*-
"""


@author: Lite Kim
"""

import category_encoders as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree

#get user input: which dataset to use
valid_dataset = False
while (not valid_dataset):
    user_in = input("Please choose a dataset <occupancy or landsat>: ")
    if (str(user_in) != 'occupancy' and str(user_in) != 'landsat'):
        print('you did not choose a valid dataset, try again')
    else:
        print('you chose:', str(user_in))
        valid_dataset = True

if (str(user_in) == 'occupancy'):
    dataset = pd.read_csv("occupancy_sensor_data.csv")
else:
    dataset = pd.read_csv("lantsat.csv", header = None)

#separate features and target
features = dataset.iloc[:,np.arange(dataset.shape[1]-1)].copy()
if (str(user_in) == 'occupancy'):
    features = features.drop(['date'], axis = 1, inplace = False) #too much variation in date, processing takes too long: drop it
target = dataset.iloc[:,dataset.shape[1]-1].copy()
features_avail = features

#choosing the features
#choose a machine learning model
#import decisiontree_script as ml_script
#ml_script.decisionTreeClassify(dataset.shape[0]+1, features_avail, target)
#import nearestneighbour_script as ml_script
#ml_script.nearestNeighbour(dataset.shape[0]+1, features_avail, target)
#import naivebayes_script as ml_script
#ml_script.naiveBayesGaussian(dataset.shape[0]+1, features_avail, target)
#import svm_script as ml_script
#ml_script.svmClassify(dataset.shape[0]+1, features_avail, target)
#import perceptron_script as ml_script
#ml_script.perceptronClassify(dataset.shape[0]+1, features_avail, target)

#tuning hyperparameters
#import decisiontree_script as ml_script
#scores = ml_script.decisionTreeOpt(features_avail, dataset.shape[0]+1, target, user_in)
#import nearestneighbour_script as ml_script
#scores = ml_script.nearestNeighbourOpt(features_avail, dataset.shape[0]+1, target, user_in)
#import naivebayes_script as ml_script
#scores = ml_script.naiveBayesCV(features_avail, dataset.shape[0]+1, target, user_in)
#print(scores.mean())
#import svm_script as ml_script
#scores = ml_script.svmOpt(features_avail, dataset.shape[0]+1, target, user_in)
#import perceptron_script as ml_script
#scores = ml_script.perceptronOpt(features_avail, dataset.shape[0]+1, target, user_in)
#results = scores.grid_scores_
#print(scores.best_score_)
#print(scores.best_params_)
#print(scores.best_estimator_)

#draw the plots
#import decisiontree_script as ml_script
import nearestneighbour_script as ml_script
#import naivebayes_script as ml_script
#import svm_script as ml_script
#import perceptron_script as ml_script
ml_script.drawPlot(features_avail, dataset.shape[0]+1, target, user_in)

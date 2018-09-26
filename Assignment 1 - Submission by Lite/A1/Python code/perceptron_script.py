# -*- coding: utf-8 -*-
"""


@author: KiwiDivo
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from linear regression demo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#PERCEPTRON CLASSIFICATION
from sklearn.linear_model import Perceptron

import feature_preparation as fp

def perceptronClassify(entries, features_avail, target):
    #initialize classification results dataframe
    clf_res = pd.DataFrame(columns = ['Feature', 'Train_Acc', 'Test_Acc'])
    #select features
    for z in range (0, features_avail.shape[1]):
        features_sel = features_avail[[features_avail.columns[z]]].copy()
        clf_res = step(features_sel, entries, target, clf_res)
        
    #for comparing maximums
    ref_acc = clf_res['Test_Acc'].max()
    #for checking available features
    ref_feat = clf_res.iloc[clf_res['Test_Acc'].idxmax()]['Feature']
    features_saved = features_avail[[ref_feat]].copy()
    features_avail = features_avail.drop([ref_feat], axis = 1, inplace = False)
    cond = True
    loop_count = 0
    
    #SELECT FEATURES
    while (features_avail.shape[1] > 0 and cond):
        #keep track of number of iterations
        loop_count += 1
        print(loop_count)
        #find the best combination of features
        clf_res = pd.DataFrame(columns = ['Feature', 'Train_Acc', 'Test_Acc'])
        for y in range (0, features_avail.shape[1]):
            features_sel = features_saved.join(features_avail[[features_avail.columns[y]]].copy())
            clf_res = step(features_sel, entries, target, clf_res)
        
        #make comparisons and check if an optimal model has been achieved
        cur_acc = clf_res['Test_Acc'].max()
        print(list(features_saved.columns.values))
        print('current reference: ', ref_acc)
        print('best with additional feature: ', cur_acc)
        #a new model is better than previous model (comparing Mean Squared Errors) - save values and reiterate
        if (ref_acc <= cur_acc):
            ref_acc = cur_acc
            ref_feat = clf_res.iloc[clf_res['Test_Acc'].idxmax()]['Feature']
            features_saved = features_saved.join(features_avail[[ref_feat]].copy())
            features_avail = features_avail.drop([ref_feat], axis = 1, inplace = False)
        #previous model is better than all new models - escape loop
        else:
            cond = False
    
    print(' ')
    print('Best combination of features:')
    print(list(features_saved.columns.values))
    print(' ')
    print(' ')
    
def step(features_sel, entries, target, clf_res):
    features_prep = fp.prepareFeatures(features_sel, entries)
    occ_x_train, occ_x_test, occ_y_train, occ_y_test = train_test_split(features_prep, target, test_size = 0.1, random_state = 0)
    
    #MODEL FITTING (classification)
    classifier = Perceptron()
    classifier.fit(occ_x_train, occ_y_train)
    occupancy_y_train_pred = classifier.predict(occ_x_train)
    occupancy_y_test_pred = classifier.predict(occ_x_test)
    
    #VERIFICATION
    train_pred_acc = 100*accuracy_score(occ_y_train, occupancy_y_train_pred)
    test_pred_acc = 100*accuracy_score(occ_y_test, occupancy_y_test_pred)
    clf_res = clf_res.append(pd.DataFrame([[features_sel.columns[features_sel.shape[1]-1], train_pred_acc, test_pred_acc]], 
                                          columns = ['Feature', 'Train_Acc', 'Test_Acc']), ignore_index = True)
    return clf_res

#HYPERPARAMETER TUNING and CROSS VALIDATION (KFOLDS)
#from sklearn.cross_validation import KFold
#from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

def perceptronOpt(features, entries, target, user_in):
    if (user_in == 'occupancy'):
        features_sel = features[['Light', 'CO2']].copy()
    else:
        features_sel = features[[20, 22, 34]].copy()
    
    features_prep = fp.prepareFeatures(features_sel, entries)
    
    classifier = Perceptron()
    aoptions = [1**-4, 1**-3, 1**-2, 1**-1, 1, 10, 100, 1000]
    param_grid = dict(alpha = aoptions)
    grid = GridSearchCV(classifier, param_grid, cv = 10, scoring = 'accuracy')
    grid.fit(features_prep, target)
    return grid

def drawPlot(features, entries, target, user_in):
    if (user_in == 'occupancy'):
        features_sel = features[['Light', 'CO2']].copy()
        classifier = Perceptron(alpha = 1.0)
    else:
        features_sel = features[[20, 22, 34]].copy()
        classifier = Perceptron(alpha = 1.0)
        
    features_prep = fp.prepareFeatures(features_sel, entries)
    classifier = classifier.fit(features_prep, target)
    pred = pd.DataFrame(classifier.predict(features_prep))
    pred = pred.iloc[:,0].copy()
    """
    asdf = pred==target
    potato = pd.Series()
    print(asdf.size)
    for i in range(0, asdf.size-10,10):
        count = 0
        for j in range(0,10):
            if (asdf[i+j]==True):
                count+=1
        pppp = pd.Series(count)
        potato = potato.append(pppp)
    return potato
    """
    valid_dataset = False
    while (not valid_dataset):
        user_in = input("Get difference plot? <y or n>: ")
        if (str(user_in) != 'y' and str(user_in) != 'n'):
            print('you did not choose a valid dataset, try again')
        else:
            valid_dataset = True
            
    if (str(user_in) == 'y'):
        plt.scatter(list(range(1,entries)), target-pred, s = 4)
    else:
        plt.scatter(list(range(1,entries)), target, s = 40, marker = (5, 0))
        plt.scatter(list(range(1,entries)), pred, s = 4)
    plt.show()
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 20:50:23 2018

@author: Junjie
"""


import time as time
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib import offsetbox


# group truth label known

# group truth label unknown
from sklearn.metrics import silhouette_score

# Contingency Matrix, unsupervised version of confusion matrix
from sklearn.metrics.cluster import contingency_matrix


random_seed = 755

# load data
#train_f = pd.read_csv("data/Occupancy_sensor/train.csv")
train = pd.read_csv("data/Occupancy_sensor/train.csv")
test = pd.read_csv("data/Occupancy_sensor/test.csv")

#train = train_f.sample(test.shape[0])

del train['index']
del test['index']
del train['date']
del test['date']

# Plot
def plot_embedding(X, y, title=None, path="data/Occupancy_sensor/"):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    #ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
        plt.savefig('%s%s.png' % (path, title))
    plt.close('all')

# Feature Selection/Extraction
def extract(dat, slide=range, max_range=None):
    data = dat.copy()
    y = data['Occupancy']
    del data['Occupancy']
    if callable(slide):
        if max_range is None:
            return y.values, data.iloc[:, slide(0, data.shape[1])].values
        if max_range is not None:
            return y.values, data.iloc[:, slide(0, max_range)].values
    if isinstance(slide ,list): 
        return y.values, data.iloc[:, list].values


col_scalling = {'Identity':None , 'ZScore':  StandardScaler}


# Cross Validation
skf  = KFold(n_splits=5, random_state=random_seed)


# Models
models = {
    "kmeans": KMeans(random_state=random_seed),
    "GMM_full": GaussianMixture( init_params='random', covariance_type = 'full', random_state=random_seed),
    "GMM_tied": GaussianMixture( init_params='random', covariance_type = 'tied', random_state=random_seed),
    "GMM_diag": GaussianMixture( init_params='random', covariance_type = 'diag', random_state=random_seed),
    "GMM_spherical": GaussianMixture( init_params='random', covariance_type = 'spherical', random_state=random_seed)   
}

# train/test 
cv = []
holdout = []
Y = train.Occupancy
y_train0, X_train0 = extract(train)
y_test0, X_test0 = extract(test)
# Hyperparameter Space
tuned_parameters = {
        "kmeans": [{'n_clusters': [2,3,4,5,6]}],
        "GMM_full": [{'n_clusters': [2,3,4,5,6]}],
        "GMM_tied": [{'n_clusters': [2,3,4,5,6]}],
        "GMM_diag": [{'n_clusters': [2,3,4,5,6]}],
        "GMM_spherical": [{'n_clusters': [2,3,4,5,6]}]
}
for sl_name, scaler in col_scalling.items():
    print("# Scaling column value by %s" % sl_name)
    if sl_name != 'Identity':
        sl = scaler()
        sl.fit(X_train0)
        X_train = sl.transform(X_train0)
        X_test = sl.transform(X_test0)
    else: 
        X_train = X_train0.copy()
        X_test = X_test0.copy()
    X_embedded = TSNE(n_components=2, init='pca').fit_transform(X_train)
    y_train = y_train0
    y_test = y_test0
    plot_embedding(X_embedded, y_train+1, title='Original %s' % (sl_name,))
    for mkey, model in models.items():
        print("# Staring fittimg model of %s" % mkey)
        print()
        
        start = time.time()
        iter_params = tuned_parameters[mkey][0]['n_clusters']
        train_list = []
        for para in iter_params:
            if mkey != 'kmeans':
                model.set_params(n_components=para)
            else:
                model.set_params(n_clusters=para)
            model.fit(X_train)
            label = model.predict(X_train)        
            plot_embedding(X_embedded, label+1, title='%s %s parameter_%d' % (sl_name, mkey, para))
            sh_score = silhouette_score(X_train, label)
            train_list.append((sl_name, mkey, para, sh_score))
        #cluster.fit(X_train,y_train)
        end = time.time()
        fitting_time = end - start
        print(fitting_time)
        train_result = pd.DataFrame(train_list, columns=['column_scale', 'model','parameter', 'score'])
        cv.append(train_result)
        
        print("Grid scores on train set:")
        print()
        best = train_result.sort_values('score', ascending=False).head(1).reset_index(drop=True)
        print(train_result)
        print("Best parameter is:")
        print(best)
        
        print("Test Set Report:")
        print()
        if mkey != 'kmeans':
            model.set_params(n_components=best['parameter'].tolist()[0])
        else:
            model.set_params(n_clusters=best['parameter'].tolist()[0])
        best_model = model.fit(X_train)
        y_true, y_pred = y_test, best_model.predict(X_test)
        print("Silhouette Score on holdout set:")
        ho_sh_score = silhouette_score(X_train, label)
        print("%0.02f" % (ho_sh_score*100.))
        print()
        best['hold_out_score'] = ho_sh_score 
        holdout.append(best)
        
output_cv = pd.concat(cv)
result = output_cv.sort_values(['column_scale', 'model','score'], ascending=False).groupby(['column_scale', 'model']).head(1).sort_values('score',ascending=False).reset_index(drop=True)

output_ho = pd.concat(holdout)
output_ho.to_csv("data/Occupancy_sensor/ho_result.csv",index=False)
# -*- coding: utf-8 -*-
"""


@author: KiwiDivo
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

#create a class to select numerical or categorical columns 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

def prepareFeatures(features_sel, entries):
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(list(features_sel))),
        ('imputer', Imputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])
    full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline)
    ])
    features_prep = pd.DataFrame(data=full_pipeline.fit_transform(features_sel),index=np.arange(1, entries))
    return features_prep
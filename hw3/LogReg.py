# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:29:36 2017

@author: Kyle
"""

from import_data import import_data
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier as RFC
import pandas
import matplotlib.pyplot as plt
import sklearn
import numpy as np
    
#Read in the Data
X, Y, Xtest, Ytest = import_data('gisette', 'gisette_train.data', 'gisette_valid.data','gisette_train.labels', 'gisette_valid.labels', head = None)
X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)  #conner.xyz  at https://stackoverflow.com/questions/20517650/how-to-delete-the-last-column-of-data-of-a-pandas-dataframe

#Normalize the Data
X = pandas.core.frame.DataFrame(preprocessing.scale(X))
Xtest = pandas.core.frame.DataFrame(preprocessing.scale(Xtest))






# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 20:10:58 2017

@author: joh10
"""

from import_data import import_data
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np

#Import Gisette
X, Y, Xtest, Ytest = import_data('gisette', 'gisette_train.data', 'gisette_valid.data','gisette_train.labels', 'gisette_valid.labels', head = None)
X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)

# standardize the data
X = preprocessing.scale(X)
Y = [0 if y <= 0 else 1 for y in np.array(Y)]
X1 = np.array(np.c_[np.ones((len(X), 1)), np.matrix(X)])


# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 13:01:27 2017

@author: Kyle
"""

from import_data import import_data
from sklearn import tree
import sklearn
import numpy as np

X, Y, Xtest, Ytest = import_data('madelon', 'madelon_train.data', 'madelon_valid.data','madelon_train.labels', 'madelon_valid.labels', head = None)
X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)  #conner.xyz  at https://stackoverflow.com/questions/20517650/how-to-delete-the-last-column-of-data-of-a-pandas-dataframe


for d in range(1,13):
    clf = tree.DecisionTreeClassifier(max_depth = d)
    clf = clf.fit(X, Y)
    
    m = clf.score(Xtest, Ytest)
    print(m)
    Ypred = clf.predict(Xtest)
    Ypred - Ytest
    print(sklearn.metrics.confusion_matrix(np.array(Ytest), Ypred), "\n")
    
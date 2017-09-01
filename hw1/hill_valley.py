# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:41:24 2017

@author: joh10
"""

# hill-valley data
from import_data import import_data
from sklearn import tree
import sklearn
import numpy as np


X, Y, Xtest, Ytest = import_data('hill_valley','X.dat', 'Xtest.dat', 'Y.dat', 'Ytest.dat', head = None)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
    
Ypred = clf.predict(Xtest)
sklearn.metrics.confusion_matrix(np.array(Ytest), Ypred)


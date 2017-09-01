# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 13:01:27 2017

@author: Kyle
"""

from import_data import import_data
from sklearn import tree
import sklearn
import numpy as np

X, Y, Xtest, Ytest = import_data('madelon', 'madelon_train.data', 'madelon_valid.data','madelon_train.labels', 'madelon_valid.labels')

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
#    
#Ypred = clf.predict(Xtest)
#sklearn.metrics.confusion_matrix(np.array(Ytest), Ypred)
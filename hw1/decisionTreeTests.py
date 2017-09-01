# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 13:01:27 2017

@author: Kyle
"""

from import_data import import_data
from sklearn import tree
import matplotlib.pyplot as plt
import sklearn
import numpy as np

X, Y, Xtest, Ytest = import_data('madelon', 'madelon_train.data', 'madelon_valid.data','madelon_train.labels', 'madelon_valid.labels', head = None)
X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)  #conner.xyz  at https://stackoverflow.com/questions/20517650/how-to-delete-the-last-column-of-data-of-a-pandas-dataframe

errorsTrain = []
errorsTest = []
D = range(1,12)
for d in D:
    clf = tree.DecisionTreeClassifier(max_depth = d)
    clf = clf.fit(X, Y)
    
    error = 1 - clf.score(X, Y)
    errorsTrain.append(error)
    
    error = 1-clf.score(Xtest, Ytest)
    errorsTest.append(error)
    
plt.plot(D, errorsTrain, label = 'Training Error')
plt.plot(D, errorsTest, label = 'Test Error')
plt.legend()
plt.show()

print('min is ', min(errorsTest), ' at d = ', np.argmin(errorsTest) + 1)

    
    

    
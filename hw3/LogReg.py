# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:29:36 2017

@author: Kyle
"""

from import_data import import_data
from sklearn import preprocessing
import pandas
import matplotlib.pyplot as plt
import sklearn
import numpy as np
    
#Read in the Data
X, Y, Xtest, Ytest = import_data('gisette', 'gisette_train.data', 'gisette_valid.data','gisette_train.labels', 'gisette_valid.labels', head = None)
X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)  #conner.xyz  at https://stackoverflow.com/questions/20517650/how-to-delete-the-last-column-of-data-of-a-pandas-dataframe

#Normalize the Data
X = preprocessing.scale(X)
Xtest = preprocessing.scale(Xtest)

#initialize variables
Nrow = len(X)
Ncol = len(X[0])
w = [0]*(Ncol + 1)

learningRate = 1
lam = .001
        

for i in range(300):
    for k in range(Ncol + 1):
        j = 0
        dL = 0
        for r in X:
            exp = np.exp(w[0] + np.dot(np.matrix(w[1:len(w)]), np.transpose(np.matrix(r))))
            dLnew = r[j]*(Y[j] - exp/(1 - exp))
            dL = dL + dLnew
            j = j + 1
        w[k] = w[k] - learningRate*lam*w[k] + learningRate*dL/Nrow
        
     
     
for k in range(1):
    j = 0
    for r in X:
#        print(np.dot(np.matrix(w[1:len(w)]), np.transpose(np.matrix(r))))
        exp = np.exp(w[0] + np.dot(np.matrix(w[1:len(w)]), np.transpose(np.matrix(r))))
        newdL = np.array(r)*(Y[j] - exp/(1 - exp))
#        dL = dL + newdL
        j = j + 1
        #for c in r:
#        i = i + 1
#        print(i)
#        print(type(r))
#        print(np.shape(r))
#        tmp = np.transpose(np.matrix(r))
#        print(type(tmp))
#        print(np.shape(tmp))
#        if i == 1:
#            break
    
    


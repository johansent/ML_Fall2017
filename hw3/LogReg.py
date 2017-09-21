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

#Add column of 1s
X1 = np.array(np.c_[np.ones((len(X), 1)), np.matrix(X)])

#initialize variables
Nrow = len(X)
Ncol = len(X1[0])
w = [0]*(Ncol)

learningRate = 1
lam = .001
        

for i in range(3):
    tmp = w[1:Ncol]
    product = np.dot(X,tmp)
    shiftedValue = w[0] + product
    expValue = np.exp(shiftedValue)
    ratio = expValue / (1 + expValue)
    error = Y[0] - ratio
   
    for k in range(Ncol):
        dLnew = X1[:,k] * error
        dL = sum(dLnew)
#        for j in range(Nrow):
##            if j == 0:
##                tmp = w[0] + np.dot(w[1:len(w)], r[1:len(r)])
##                print(tmp)
#            #e_dotprod = np.exp(w[0] + np.dot(tmp, r[1:Nrow]))
##            print(np.shape(w))
#            dLnew = r[k]*(Y[0][j] - e_dotprod/(1 + e_dotprod))
##            print(np.shape(w))
#            dL = dL + dLnew
        w[k] = w[k] - learningRate*lam*w[k] + learningRate*dL/Nrow
    print(i)
        
     
     

    
    


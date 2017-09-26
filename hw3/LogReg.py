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

def TrainWeights(X,Y,k):
    X = preprocessing.scale(X)
    Y = [0 if y <= 0 else 1 for y in np.array(Y)]
    X1 = np.array(np.c_[np.ones((len(X), 1)), np.matrix(X)])
    #initialize variables
    Nrow = len(X)
    Ncol = len(X1[0])
    w = np.array([0]*(Ncol))

    learningRate = .1
    lam = .001
        

    for i in range(k):
        tmp = w[1:Ncol]
        product = np.dot(X,tmp)
        #print('product', product)
        shiftedValue = w[0] + product
        #print('shiftedValue',shiftedValue)
        expValue = np.exp(shiftedValue)
        #print('expValue', expValue)
        ratio = expValue / (1 + expValue)
        #print('ratio',ratio)
        error = Y - ratio
        #print('error',error)
        
        
        dLnew = np.dot(np.transpose(X1),error)
        #print('dLnew',dLnew)
        w = w + learningRate * (-lam*w + dLnew/Nrow)
        
#        dL = np.sum(dLnew)
#        print('Dl', dL)
#        w = w - learningRate * (lam*w + dL/Nrow)
           
        #for k in range(Ncol):
            #dL = sum(dLnew)
#        for j in range(Nrow):
##            if j == 0:
##                tmp = w[0] + np.dot(w[1:len(w)], r[1:len(r)])
##                print(tmp)
#            #e_dotprod = np.exp(w[0] + np.dot(tmp, r[1:Nrow]))
##            print(np.shape(w))
#            dLnew = r[k]*(Y[0][j] - e_dotprod/(1 + e_dotprod))
##            print(np.shape(w))
#            dL = dL + dLnew
            #w[k] = w[k] - learningRate*lam*w[k] + learningRate*dL/Nrow
        #print(w[1])
    return w

def Test(w, X, Y):
    #Add column of 1s
    X = preprocessing.scale(X)
    X = np.array(np.c_[np.ones((len(X), 1)), np.matrix(X)])
    Y = [0 if y <= 0 else 1 for y in np.array(Y)]
    wx = 1/(1+np.exp(-1 * np.dot(X, w)))
    Ypredict = [0 if x < .5 else 1 for x in wx]
    
    
    results = np.array(Y) - np.array(Ypredict)
    return sum(abs(results))/len(Y)
    
    

##Gisette Data
#    
##Read in the Data
#X, Y, Xtest, Ytest = import_data('gisette', 'gisette_train.data', 'gisette_valid.data','gisette_train.labels', 'gisette_valid.labels', head = None)
#X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
#Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)  #conner.xyz  at https://stackoverflow.com/questions/20517650/how-to-delete-the-last-column-of-data-of-a-pandas-dataframe
#
##Normalize the Data
#X = preprocessing.scale(X)
#Xtest = preprocessing.scale(Xtest)
#
#
#
#w = TrainWeights(X,Y)
##print(w)

#Arcene Test
X, Y, Xtest, Ytest = import_data('arcene', 'arcene_train.data', 'arcene_valid.data','arcene_train.labels', 'arcene_valid.labels', head = None)
X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)



w = TrainWeights(X,Y,2)
print(w)
w = TrainWeights(X,Y,300)
print(w)

print(Test(w,Xtest, Ytest))
#Plot(K,eTest,eTrain,'Arcene')

#min_table['Arcene      '] = [min(eTest), K[np.argmin(eTest)]]
     
     

    
    


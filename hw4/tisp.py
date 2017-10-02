# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 20:10:58 2017

@author: joh10
"""

from import_data import import_data
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np

def updateWeights(X,X1,Y,w, Ncol,Nrow, learningRate, lam, thresh):
    tmp = w[1:Ncol]
    product = np.dot(X,tmp)
    shiftedValue = w[0] + product
    expValue = np.exp(shiftedValue)
    ratio = expValue / (1 + expValue)
    error = Y - ratio
    dLnew = np.dot(np.transpose(X1),error)
    w = w + learningRate * (-lam*w + dLnew/Nrow)
    w = (w) * (abs(w) >= thresh)
    #print(sum())
    return w

def Test(w, X, Y):
    #nomralize data
    X = preprocessing.scale(X)
    #Add column of 1s
    X = np.array(np.c_[np.ones((len(X), 1)), np.matrix(X)])
    Y = [0 if y <= 0 else 1 for y in np.array(Y)]
    wx = 1/(1+np.exp(-1 * np.dot(X, w)))
    Ypredict = [0 if x < .5 else 1 for x in wx]
    results = np.array(Y) - np.array(Ypredict)
    return sum(abs(results))/len(Y)
    
def Plot(x,y1,y2,title,legendLoc = 1):
    plt.title(title)
    plt.plot(x,y1,label = 'Test Error')
    plt.plot(x,y2,label = 'Training Error')
    plt.legend(loc = legendLoc)
    plt.xlabel('iteration count')
    plt.ylabel('Misclassification Error')
    plt.show()
    
def TrainWeights(X,Y,Xtest,Ytest,k, learnRate = .01, thresh = .001):
    X = preprocessing.scale(X)
    Y = [0 if y <= 0 else 1 for y in np.array(Y)]
    X1 = np.array(np.c_[np.ones((len(X), 1)), np.matrix(X)])
    #initialize variables
    Nrow = len(X)
    Ncol = len(X1[0])

    learningRate = learnRate
    lam = .001
    
    w = np.array([0]*(Ncol))
    
    y1 = []
    y2 = []
    for i in range(k):
        w = updateWeights(X,X1,Y,w, Ncol,Nrow, learningRate, lam, thresh)
        y1.append(Test(w,Xtest, Ytest))
        y2.append(Test(w,X,Y))
        if(i%10 == 0):
            print(sum(abs(w) >= thresh))
        
    return y1,y2


import timeit
start = timeit.default_timer()
min_table = {}

#Gisette Data
#10:
#30:(48  .02)
#100
#300:.01
    
#Read in the Data
X, Y, Xtest, Ytest = import_data('gisette', 'gisette_train.data', 'gisette_valid.data','gisette_train.labels', 'gisette_valid.labels', head = None)
X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)  #conner.xyz  at https://stackoverflow.com/questions/20517650/how-to-delete-the-last-column-of-data-of-a-pandas-dataframe

niter = 100
y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter,.1, .02)

Plot(range(niter), y1, y2, 'Gisette Errors')
min_table['Gisette'] = [min(y1), min(y2)]

#Arcene Test
#10
#30
#100
#300
X, Y, Xtest, Ytest = import_data('arcene', 'arcene_train.data', 'arcene_valid.data','arcene_train.labels', 'arcene_valid.labels', head = None)
X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)

niter = 100
y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter, .001, .0001)    

Plot(range(niter), y1, y2, 'Arcene Errors')
#Plot(K,eTest,eTrain,'Arcene')
min_table['Arcene'] = [min(y1), min(y2)]

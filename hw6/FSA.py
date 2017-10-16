# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:29:36 2017

@author: Kyle
"""


from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath('../'))
from import_data import import_data

#def normalize(X, Xtest):
#    npX = np.array(X)
#    mns = np.mean(npX, axis = 0)
#    std = np.std(npX, axis = 0)
#    Xnew = (npX - mns)/std
#    Xtest_new = (np.array(Xtest) - mns)/std
#    
#    return Xnew, Xtest_new

def updateWeights(X,X1,Y,w, Ncol,Nrow, learningRate, lam):
    tmp = w[1:Ncol]
    product = np.dot(X,tmp)
    shiftedValue = w[0] + product
    expValue = np.exp(shiftedValue)
    ratio = expValue / (1 + expValue)
    error = Y - ratio
    dLnew = np.dot(np.transpose(X1),error)
    w = w + learningRate * (-lam*w + dLnew/Nrow)
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
    
def TrainWeights(X,Y,Xtest,Ytest,k, learnRate = .01):
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
        w = updateWeights(X,X1,Y,w, Ncol,Nrow, learningRate, lam)
        y1.append(Test(w,Xtest, Ytest))
        y2.append(Test(w,X,Y))
        
    return y1,y2
    


### Script
import timeit
start = timeit.default_timer()
min_table = {}

#Gisette Data
    
#Read in the Data
X, Y, Xtest, Ytest = import_data('gisette', 'gisette_train.data', 'gisette_valid.data','gisette_train.labels', 'gisette_valid.labels', head = None)
X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)  #conner.xyz  at https://stackoverflow.com/questions/20517650/how-to-delete-the-last-column-of-data-of-a-pandas-dataframe

niter = 50
y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter,.1)

Plot(range(niter), y1, y2, 'Gisette Errors')
min_table['Gisette'] = [min(y1), min(y2)]


#Arcene Test
X, Y, Xtest, Ytest = import_data('arcene', 'arcene_train.data', 'arcene_valid.data','arcene_train.labels', 'arcene_valid.labels', head = None, norm = True)
X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)

niter = 100
y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter, .001)    

Plot(range(niter), y1, y2, 'Arcene Errors')
#Plot(K,eTest,eTrain,'Arcene')
min_table['Arcene'] = [min(y1), min(y2)]

# Madelon
X, Y, Xtest, Ytest = import_data('madelon', 'madelon_train.data', 'madelon_valid.data','madelon_train.labels', 'madelon_valid.labels', head = None)
X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)

niter = 500
y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter)

Plot(range(niter), y1, y2, 'Madelon Errors', 3)
min_table['Madelon'] = [min(y1), min(y2)]


# Hill/Valley
X, Y, Xtest, Ytest = import_data('hill_valley', 'X.dat', 'Xtest.dat','Y.dat', 'Ytest.dat', head = None)

niter = 3000
y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter)

Plot(range(niter), y1, y2, 'Hill/Valley Errors', 3)
min_table['Hill/Valley'] = [min(y1), min(y2)]

#Summary Of Results
print('{0:17s} {1:12s} {2:12s}'.format('Data name', 'Test Error', 'Training Error'))
print('---------------------------------------')
for k in min_table.keys():
    print('{0:12s} {1:13f} {2:12f}'.format(k, min_table[k][0], min_table[k][1]))
    
stop = timeit.default_timer()
print(stop - start)

     
     

    
    


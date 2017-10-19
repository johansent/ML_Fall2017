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

def updateWeights(X,Y,w, Ncol,Nrow, learningRate, s):
    #tmp = w[1:Ncol]
#    product = np.dot(X,w)
#    divisor = 2 + product*product - 2 * np.transpose(Y * np.transpose(product))
#    Value = 2 * product * X 

    sumation = [0] * Ncol
    #print(sum(X))
    for k, Row in X.iterrows():
        row = np.array(Row)
#        for r in row:
#            print(r, type(r))
        #print('row', np.sort(row))
        #print('w',type(w[-1]))
        #print(sum(w), sum(row))
        product = Y[k] * np.dot(w,row)
        #print(np.dot(w,row))
        if(product < 1):
            divisor = 2 + product*product - 2 * product
            numerator = 2 * (product - 1)
            value = (numerator / divisor) * row
            sumation = sumation + value
        
    #print(type(sumation), type(s), type(w))
    derivative = sumation + s * w
    w = w - learningRate * derivative
    #print('w',w)
    
    

    
    return w
    
        
    
    
#    shiftedValue = w[0] + product
#    expValue = np.exp(shiftedValue)
#    ratio = expValue / (1 + expValue)
#    error = Y - ratio
#    dLnew = np.dot(np.transpose(X1),error)
#    w = w + learningRate * (-lam*w + dLnew/Nrow)
#    return w

def Test(w, X, Y):
    #Add column of 1s
    #X = np.array(np.c_[np.ones((len(X), 1))X, np.matrix(X)])
    Y = [0 if y <= 0 else 1 for y in np.array(Y)]
    #wx = 1/(1+np.exp(-1 * np.dot(X, w)))
    wx = np.dot(X,w)
    Ypredict = [0 if x < 0 else 1 for x in wx]
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
    
def TrainWeights(X,Y,Xtest,Ytest,niter,k, learnRate = .01):
    Y = [-1 if y <= 0 else 1 for y in np.array(Y)]
    #X1 = np.array(np.c_[np.ones((len(X), 1)), np.matrix(X)])
    #initialize variables
    Nrow = len(X)
    Ncol = len(X.columns)

    s = .001
    
    w = np.array([0]*(Ncol))
    
    testErrors = []
    trainingErrors = []
    for i in range(niter):
        Mi = getMi(Ncol, k, niter, i, u = 100)
        #print('Mi', Mi)
        w = updateWeights(X,Y,w, Ncol,Nrow, learnRate, s)
        w,X,Xtest,Ncol = getMBest(w, X, Xtest, Mi, Ncol)
        testErrors.append(Test(w,Xtest, Ytest))
        trainingErrors.append(Test(w,X,Y))
        #print(testErrors)
        
    return testErrors,trainingErrors

def getMi(M, k, N, i, u = 100):
    return round(k + (M - k) * max([0,(N - 2 * i)/(2 * i * u + N)]))
    
def getMBest(w, X, Xtest, M, Ncol):
    best = sorted(range(len(w)), key=lambda i: w[i])[-M:]
    worst = sorted(range(len(w)), key = lambda i: w[i])[0:(Ncol - M)]
    w = np.array([w[i] for i in best])
    X.drop(X.columns[worst], axis=1, inplace=True)
    Xtest.drop(Xtest.columns[worst], axis=1, inplace=True)
    return w, X, Xtest, len(X.columns)
    


### Script
import timeit
start = timeit.default_timer()
min_table = {}

#Gisette Data
    
#Read in the Data
X, Y, Xtest, Ytest = import_data('gisette', 'gisette_train.data', 'gisette_valid.data','gisette_train.labels', 'gisette_valid.labels', head = None, norm = True, removeCol = True)

niter = 100
y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter,10,.1)

Plot(range(niter), y1, y2, 'Gisette Errors')
min_table['Gisette'] = [min(y1), min(y2)]

#
##Arcene Test
#X, Y, Xtest, Ytest = import_data('arcene', 'arcene_train.data', 'arcene_valid.data','arcene_train.labels', 'arcene_valid.labels', head = None, norm = True, removeCol = True)
#
#niter = 500
#y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter,100, .001)    
#
#Plot(range(niter), y1, y2, 'Arcene Errors')
##Plot(K,eTest,eTrain,'Arcene')
#min_table['Arcene'] = [min(y1), min(y2)]

## Madelon
#X, Y, Xtest, Ytest = import_data('madelon', 'madelon_train.data', 'madelon_valid.data','madelon_train.labels', 'madelon_valid.labels', head = None, norm = True, removeCol = True)
#
#niter = 500
#y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter, 100, .001)
#
#Plot(range(niter), y1, y2, 'Madelon Errors', 3)
#min_table['Madelon'] = [min(y1), min(y2)]

#
## Hill/Valley
#X, Y, Xtest, Ytest = import_data('hill_valley', 'X.dat', 'Xtest.dat','Y.dat', 'Ytest.dat', head = None)
#
#niter = 3000
#y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter)
#
#Plot(range(niter), y1, y2, 'Hill/Valley Errors', 3)
#min_table['Hill/Valley'] = [min(y1), min(y2)]


#Summary Of Results
print('{0:17s} {1:12s} {2:12s}'.format('Data name', 'Test Error', 'Training Error'))
print('---------------------------------------')
for k in min_table.keys():
    print('{0:12s} {1:13f} {2:12f}'.format(k, min_table[k][0], min_table[k][1]))
    
stop = timeit.default_timer()
print(stop - start)

     
     

    
    


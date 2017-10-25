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

def updateWeights(X,Y,w, Ncol,Nrow, learningRate, s):
    y = np.array(Y)
    L = len(w)
    summation = [] #np.array([[0]*(Ncol)]*L)
    lorenz = 0
    for k, Row in X.iterrows():
        row = np.array(Row)
        product_y = np.dot(w[y[k]],row)
        sumj = []
        for j in range(L):
            vec = [0] * Ncol #stays zero unless y != j and prod <= 0
            if y[k] != j:
                product_j = np.dot(w[j],row)
                prod = product_y - product_j - 1
                if prod <= 0:
                    vec = - (2 * prod / (1 + prod*prod)) * row
                    lorenz += np.log(1 + (prod - 1)**2)
            sumj.append(vec)
	
        if k == 0:
            summation = np.array(sumj)
        else:
            summation = summation + sumj
            #summation = (k > 0)*summation + sumj # summation is sumj on first iteration

			
            #lorenz.append(np.log(1 + (product - 1)**2))
            #else:
            #lorenz.append(0)
        
        derivative = summation + (s * w)
        w = w - (learningRate * derivative)    
    
    loss = -(lorenz + L* s * np.linalg.norm(w, 'fro'))
    
    return w, loss
    
        
    
    
#    shiftedValue = w[0] + product
#    expValue = np.exp(shiftedValue)
#    ratio = expValue / (1 + expValue)
#    error = Y - ratio
#    dLnew = np.dot(np.transpose(X1),error)
#    w = w + learningRate * (-lam*w + dLnew/Nrow)
#    return w

def Test(w, X, Y):
    #Y = [0 if y <= 0 else 1 for y in np.array(Y)]
    y = np.array(Y)
    wx = np.dot(X,np.transpose(w))
    Ypredict = [np.argmax(x) for x in wx]
    #print('ypred', np.shape(np.array(Ypredict)))
    #print('y', np.shape(np.array(Y)))
    results = [y[i][0] - Ypredict[i] for i in range(len(Ypredict))] #np.array(Y) - Ypredict
    #print('results', np.shape(results))
    results = np.array([1 if x != 0 else 0 for x in results])
    return sum(abs(results))/len(Ypredict)
    
def Plot(x,y1,y2,title,legendLoc = 1, labels = ['iteration count', 'Misclassification Error']):
    plt.title(title)
    if y2 == None:
        plt.plot(x,y1)
    else:
        plt.plot(x,y1,label = 'Test Error')
        plt.plot(x,y2,label = 'Training Error')
        plt.legend(loc = legendLoc)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()
    
def TrainWeights(X,Y,Xtest,Ytest,niter,k, learnRate = .01):
    #Y = [-1 if y <= 0 else 1 for y in np.array(Y)]
    Nrow = len(X)
    Ncol = len(X.columns)
    L = 7

    s = .001
    
    w = np.array([[0]*(Ncol)]*L)
    
    loss = []
    testErrors = []
    trainingErrors = []
    for i in range(niter):
        
        #print('Mi', Mi)
        w, newloss = updateWeights(X,Y,w, Ncol,Nrow, learnRate, s)
        loss.append(newloss)
        if Ncol > k:
            Mi = getMi(Ncol, k, niter, i, u = 100)
            w,X,Xtest,Ncol = getMBest(w, X, Xtest, Mi, Ncol)
        testErrors.append(Test(w,Xtest, Ytest))
        trainingErrors.append(Test(w,X,Y))
        
        if(i % 10 == 0):
            print('i', i)
        
    return testErrors,trainingErrors, loss

def getMi(M, k, N, i, u = 100):
    return round(k + (M - k) * max([0,(N - 2 * i)/(2 * i * u + N)]))
    
def getMBest(w, X, Xtest, M, Ncol):
    #print(type(w))
    summation = sum(w)
    #print(len(summation))
    best = sorted(range(len(summation)), key=lambda i: summation[i])[-M:]
    worst = sorted(range(len(summation)), key = lambda i: summation[i])[0:(Ncol - M)]
    w = np.array([[x[i] for i in best] for x in w])
    X.drop(X.columns[worst], axis=1, inplace=True)
    Xtest.drop(Xtest.columns[worst], axis=1, inplace=True)
    #print(np.shape(X))
    return w, X, Xtest, len(X.columns)
    
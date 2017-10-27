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
    #y = np.array(Y)
    L = len(w)
    derivative = np.array([[0.0]*(Ncol)]*L)
    lorenz = 0
    X = np.array(X)
    product_Y = np.dot(X, np.transpose(w))
    total = 0
    totals = 0
    #print('pY,', product_Y)
    for k in range(Nrow):
        #row = np.array(Row)
        product_y = product_Y[k][Y[k]]#np.dot(w[Y[k]],row)
        summation = np.array([[0.0]*(Ncol)]*L)
        #sumj = []
        for j in range(L):
            #vec = [0] * Ncol #stays zero unless y != j and prod <= 0
            if Y[k] != j:
                product_j = product_Y[k][j]#np.dot(w[j],row)
                prod = product_y - product_j - 1
                #print('prod',np.shape(prod))
                #print('py', product_y)
                if prod <= 0:
                    #print(summation[j], - (2 * prod / (1 + prod*prod)) * X[k])
                    summation[j] = summation[j] - (((2 * prod) / (1 + prod*prod)) * X[k])#row
                    #print(np.shape(- ((2 * prod) / (1 + prod*prod))))
                    #print('X[k]', np.shape(X[k]))
                    lorenz += np.log(1 + (prod)**2)
                    
                    if j == 0:
                        total +=  - (((2 * prod) / (1 + prod*prod)) *X[k])[0]
                        totals += summation[0][0]
            #sumj.append(vec)
	
#        if k == 0:
#            summation = np.array(sumj)
#        else:
        #summation = summation + sumj
            #summation = (k > 0)*summation + sumj # summation is sumj on first iteration

			
            #lorenz.append(np.log(1 + (product - 1)**2))
            #else:
            #lorenz.append(0)
        
        derivative += summation 
    w = w - (learningRate * (derivative+ (s * w))) 
    #print('w', w[0][0])
    #print('derivative', derivative[0][0])
    #print('total', total, 'total-s', totals)
    
    loss = (lorenz + s * np.linalg.norm(w, 'fro'))
    
    return w, loss

def updateWeights1(X,Y,w, Ncol,Nrow, learningRate, s):
    #y = np.array(Y)
    L = len(w)
    #summation = np.array([[0]*(Ncol)]*L)
    derivative = np.array([[0.0]*(Ncol)]*L)
    lorenz = 0.0
    X = np.array(X)
    U = np.dot(X, np.transpose(w))
    Uy = [U[i][Y[i]] for i in range(Nrow)]
    #print('pY,', product_Y)
    for l in range(L):
        diff = (Uy - U[:,l]) - 1
        diff = np.array([-((2 * d) / (1 + d*d)) if d <= 0 else 0 for d in diff])
        logical = [l != Y[i] for i in range(Nrow)]
        diff = diff * logical
        lorenz += sum(np.log(1 + diff**2))
        dmat = diff * X.transpose()
        #print('l', l, '  dmat', np.shape(dmat), np.sum(dmat, axis = 1)[0])
        derivative[l] = np.sum(dmat, axis = 1)
        #print('dl', np.shape(derivative[l]))
    total = [X[i][0] if logical[i] else 0 for i in range(Nrow)] 
    #print('total', np.shape(total), sum(total))
        
    w = w - (learningRate * (derivative + (s * w))) 
    #print('w', w[0][0])
    #print('derivative', derivative[0][0])
    loss = (lorenz + s * np.linalg.norm(w, 'fro'))
    
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
    #y = np.array(Y)
    wx = np.dot(X,np.transpose(w))
    Ypredict = [np.argmax(x) for x in wx]
    #print('ypred', np.shape(np.array(Ypredict)))
    #print('y', np.shape(np.array(Y)))
    #results = [y[i][0] - Ypredict[i] for i in range(len(Ypredict))] #np.array(Y) - Ypredict
    results = Y - np.array(Ypredict)
    #print('results', np.shape(results))
    results = np.array([1 if x != 0 else 0 for x in results])
    return sum(results)/len(Ypredict)
    
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
    Y = np.array(Y)[:,0]
    Ytest = np.array(Ytest)[:,0]
    #print(np.shape(Y))
    X1 = X.copy()
    Xtest1 = Xtest.copy()

    s = .001
    
    w = np.array([[0]*(Ncol)]*L)
    
    loss = []
    testErrors = []
    trainingErrors = []
    for i in range(niter):
        
        #print('Mi', Mi)
        w, newloss = updateWeights(X1,Y,w, Ncol,Nrow, learnRate, s)
        loss.append(newloss)
        if Ncol > k:
            Mi = getMi(Ncol, k, niter, i, u = 100)
            w,X,Xtest,Ncol = getMBest(w, X1, Xtest1, Mi, Ncol)
        testErrors.append(Test(w,Xtest1, Ytest))
        trainingErrors.append(Test(w,X1,Y))
        
        if(i % 50 == 0):
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
    w = np.array([[x[i] for i in sorted(best)] for x in w])
    X.drop(X.columns[worst], axis=1, inplace=True)
    Xtest.drop(Xtest.columns[worst], axis=1, inplace=True)
    #print(np.shape(X))
    return w, X, Xtest, len(X.columns)
    
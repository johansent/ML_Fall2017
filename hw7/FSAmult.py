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

    summation = []
    lorenz = []
    for k, Row in X.iterrows():
        row = np.array(Row)
		product_y = np.dot(w[Y[k]],row)
		sumj = []
		for j in range(len(w[0])):
			vec = [0] * Ncol #stays zero unless y != j and prod <= 0
			if Y[k] != j:
				product_j = np.dot(w[j],row)
				prod = product_y - product_j - 1
				if prod <= 0:
					vec = - (2 * prod / (1 + prod*prod)) * row
			sumj.append(vec)
	
		if k == 0:
			summation = sumj
		else:
			summation = summation + sumj
		#summation = (k > 0)*summation + sumj # summation is sumj on first iteration

			
			#lorenz.append(np.log(1 + (product - 1)**2))
		#else:
			#lorenz.append(0)
        
	derivative = summation + (s * w)
	w = w - (learningRate * derivative)    
    
    #loss = -(sum(lorenz) + s * np.dot(w, w))
    
    return w, loss
    
        
    
    
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
    Y = [-1 if y <= 0 else 1 for y in np.array(Y)]
    #X1 = np.array(np.c_[np.ones((len(X), 1)), np.matrix(X)])
    #initialize variables
    Nrow = len(X)
    Ncol = len(X.columns)

    s = .001
    
    w = np.array([0]*(Ncol))
    
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
        
        #if i == 450 or i == 451 or i == 452:
            #print(w)
        #print(testErrors)
        
    return testErrors,trainingErrors, loss

def getMi(M, k, N, i, u = 100):
    return round(k + (M - k) * max([0,(N - 2 * i)/(2 * i * u + N)]))
    
def getMBest(w, X, Xtest, M, Ncol):
    best = sorted(range(len(w)), key=lambda i: w[i])[-M:]
    worst = sorted(range(len(w)), key = lambda i: w[i])[0:(Ncol - M)]
    w = np.array([w[i] for i in best])
    X.drop(X.columns[worst], axis=1, inplace=True)
    Xtest.drop(Xtest.columns[worst], axis=1, inplace=True)
    #print(np.shape(X))
    return w, X, Xtest, len(X.columns)
    


### Script
#import timeit
#start = timeit.default_timer()
#min_table = {}
#
##Gisette Data
#    
##Read in the Data
#X, Y, Xtest, Ytest = import_data('gisette', 'gisette_train.data', 'gisette_valid.data','gisette_train.labels', 'gisette_valid.labels', head = None, norm = True, removeCol = True)
#
#niter = 500
#y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter,100,.001)
#
#Plot(range(niter), y1, y2, 'Gisette Errors')
#min_table['Gisette'] = [min(y1), min(y2)]

#
##Arcene Test
#X, Y, Xtest, Ytest = import_data('arcene', 'arcene_train.data', 'arcene_valid.data','arcene_train.labels', 'arcene_valid.labels', head = None, norm = True, removeCol = True)
#
#niter = 500
#y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter,20, .001)    
#
#Plot(range(niter), y1, y2, 'Arcene Errors')
##Plot(K,eTest,eTrain,'Arcene')
#min_table['Arcene'] = [min(y1), min(y2)]

## Madelon
#X, Y, Xtest, Ytest = import_data('madelon', 'madelon_train.data', 'madelon_valid.data','madelon_train.labels', 'madelon_valid.labels', head = None, norm = True, removeCol = True)
#
#niter = 500
#y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter, 10, .001)
#
#Plot(range(niter), y1, y2, 'Madelon Errors', 3)
#min_table['Madelon'] = [min(y1), min(y2)]

#
## Hill/Valley
#X, Y, Xtest, Ytest = import_data('hill_valley', 'X.dat', 'Xtest.dat','Y.dat', 'Ytest.dat', head = None, norm = True)
#
#niter = 50
#y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter, 10, .01)
#
#Plot(range(niter), y1, y2, 'Hill/Valley Errors', 3)
#min_table['Hill/Valley'] = [min(y1), min(y2)]


##Summary Of Results
#print('{0:17s} {1:12s} {2:12s}'.format('Data name', 'Test Error', 'Training Error'))
#print('---------------------------------------')
#for k in min_table.keys():
#    print('{0:12s} {1:13f} {2:12f}'.format(k, min_table[k][0], min_table[k][1]))
#    
#stop = timeit.default_timer()
#print(stop - start)

     
     

    
    


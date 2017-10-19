# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 20:10:58 2017

@author: joh10
"""


from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath('../'))
from import_data import import_data

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
    #X = preprocessing.scale(X)
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
    plt.xlabel('Number of Features')
    plt.ylabel('Misclassification Error')
    plt.show()
    
def TrainWeights(X,Y,Xtest,Ytest,k, learnRate = .01, thresh = .001):
    #X = preprocessing.scale(X)
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

##Gisette Data
##10:
##30:(48  .02)
##100
##300:.01
#    
##Read in the Data
#X, Y, Xtest, Ytest = import_data('gisette', 'gisette_train.data', 'gisette_valid.data','gisette_train.labels', 'gisette_valid.labels', head = None)
#X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
#Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)  #conner.xyz  at https://stackoverflow.com/questions/20517650/how-to-delete-the-last-column-of-data-of-a-pandas-dataframe
#
#niter = 100
#y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter,.1, .02)
#
#Plot(range(niter), y1, y2, 'Gisette Errors')
#min_table['Gisette'] = [min(y1), min(y2)]
## guess and check to get the numbers for the plot below
#Plot([11, 31, 101, 299], [.144, .133, .084, .026], [.1317, .126, .0735, .0313], 'Gisette Errors', 3)

#Arcene Test
#10 (10, 0.00021) {'Arcene': [0.34000000000000002, 0.31]}
#30 (23, 0.0002) {'Arcene': [0.31, 0.26000000000000001]}
#100 (92, 0.00018) {'Arcene': [0.27000000000000002, 0.27000000000000002]}
#300 (308, 0.000163) {'Arcene': [0.28999999999999998, 0.34000000000000002]}
X, Y, Xtest, Ytest = import_data('arcene', 'arcene_train.data', 'arcene_valid.data','arcene_train.labels', 'arcene_valid.labels', head = None, removeCol = True, norm = True)
#X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
#Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)

niter = 100
y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter, .001, .0001)    

Plot(range(niter), y1, y2, 'Arcene Errors')
##Plot(K,eTest,eTrain,'Arcene')
min_table['Arcene'] = [min(y1), min(y2)]


## Hill/Valley
##10: (10, 0.00000979) {'Hill/Valley': [0.49834983498349833, 0.48349834983498352]}
##30: (23, 0.000009) {'Hill/Valley': [0.49669966996699672, 0.48349834983498352]}
##100: (101, 0.000001) {'Hill/Valley': [0.49504950495049505, 0.48184818481848185]}
##300: NA
#X, Y, Xtest, Ytest = import_data('hill_valley', 'X.dat', 'Xtest.dat','Y.dat', 'Ytest.dat', head = None)
#
#niter = 100
#y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter, 0.001, 0.0000003)
#
#Plot(range(niter), y1, y2, 'Hill/Valley Errors', 3)
#min_table['Hill/Valley'] = [min(y1), min(y2)]



## Madelon
##10: (11, 0.00005) {'Madelon': [0.40999999999999998, 0.40500000000000003]}
##30: (31, 0.000024) {'Madelon': [0.40500000000000003, 0.38300000000000001]}
##100: (101, 0.000016) {'Madelon': [0.40333333333333332, 0.36399999999999999]}
##300: (299, 0.0000065) {'Madelon': [0.41499999999999998, 0.34849999999999998]}
#X, Y, Xtest, Ytest = import_data('madelon', 'madelon_train.data', 'madelon_valid.data','madelon_train.labels', 'madelon_valid.labels', head = None)
#X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)
#Xtest.drop(Xtest.columns[len(Xtest.columns)-1], axis=1, inplace=True)
#
#niter = 100
#y1, y2 = TrainWeights(X,Y,Xtest,Ytest,niter, 0.001, 0.0000065)
#
#Plot(range(niter), y1, y2, 'Madelon Errors', 2)
#min_table['Madelon'] = [min(y1), min(y2)]
print(min_table)

#Plot([14, 35, 108, 300], [.144, .133, .084, .026], [.1317, .126, .0735, .0313], 'Gisette Errors', 3)
#Plot([10, 23, 92, 308], [.34, .31, .27, .29], [.31, .26, .27, .34], 'Arcene Errors', 2)
#Plot([11, 31, 101, 299], [.41, .405, .403333333, .415], [.405, .383, .364, .3485], 'Madelon Errors', 3)